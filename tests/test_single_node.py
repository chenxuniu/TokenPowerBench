import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tokenpowerbench import api, cli
from tokenpowerbench.energy import EnergyMetrics


class FakeMonitor:
    def __init__(self):
        self.capabilities = {"gpu": {"available": True}, "cpu": {"available": False}}
        self.samples = {"gpu": [{"timestamp_s": 1.0, "power_w": [300.0]}]}
        self.closed = False
        self.stopped = False
        self.windows = []

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def compute_metrics(self, duration, tokens, responses, start_time=None, end_time=None):
        self.windows.append((start_time, end_time, tokens))
        return EnergyMetrics(duration=duration, total_output_tokens=tokens, num_responses=responses)


class FakeEngine:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def setup_model(self, *args, **kwargs):
        return self

    def run_inference(self, *args, **kwargs):
        return [object()]

    def run_benchmark(self, prompts, *args):
        return [object() for p in prompts], 1.0, 5.0

    def estimate_tokens(self, outputs):
        return 3 * len(outputs)


class SingleNodeTests(unittest.TestCase):
    def test_reject_concurrent_phase_measurement(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            cli.parse_args(["--model", "test", "--phase-profiling", "--batch-sizes", "2"])
        self.assertEqual(exc.exception.code, 2)

    def test_invalid_numeric_configuration(self):
        for args in (["--batch-sizes", "0"], ["--temperature", "nan"], ["--num-samples", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(["--model", "test", *args])

    def test_monitor_check_does_not_load_model(self):
        monitor = FakeMonitor()
        with patch.object(api, "create_monitor", return_value=monitor), patch.object(api, "VLLMEngine") as engine, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--check-monitor", "--monitor", "gpu_only"]), 0)
        engine.assert_not_called()
        self.assertTrue(monitor.closed)

    def test_identity_reported_even_if_sensor_initialization_fails(self):
        output = io.StringIO()
        with patch.object(cli, "runtime_identity", return_value={"euid": 0, "is_root": True}), patch.object(api, "create_monitor", side_effect=RuntimeError("no GPU")), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--check-monitor"]), 1)
        self.assertTrue(json.loads(output.getvalue())["runtime"]["is_root"])

    def test_explicit_parallelism_must_match_monitored_devices(self):
        monitor = FakeMonitor()
        monitor.capabilities["gpu"]["devices"] = [{"index": 0}, {"index": 1}]
        with patch.object(api, "create_monitor", return_value=monitor), patch.object(api, "VLLMEngine") as engine, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--model", "test", "--tensor-parallel-size", "1"]), 1)
        engine.assert_not_called()
        self.assertTrue(monitor.closed)

    def test_phase_windows_and_decode_token_denominator(self):
        monitor = FakeMonitor()
        events = [{"request_id": "r1", "submitted_s": 0.0, "dispatch_completed_s": 1.0, "prefill_start_s": 0.0,
                   "first_token_s": 3.0, "finished_s": 8.0, "output_tokens": 6}]
        api.phase_results(monitor, events)
        self.assertEqual(monitor.windows, [(0.0, 3.0, 0), (3.0, 8.0, 5)])

    def test_one_token_has_no_decode_energy(self):
        monitor = FakeMonitor()
        events = [{"prefill_start_s": 1.0, "first_token_s": 3.0, "finished_s": 3.0, "output_tokens": 1}]
        result = api.phase_results(monitor, events)[0]
        self.assertEqual(result["decode_energy"]["status"], "empty_window")
        self.assertEqual(len(monitor.windows), 1)

    def run_fixture(self, engine, expect_status):
        monitor = FakeMonitor()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            prompts = base / "input.json"
            prompts.write_text('["prompt one", "prompt two"]')
            with patch.object(api, "create_monitor", return_value=monitor), patch.object(api, "VLLMEngine", return_value=engine), patch.object(api, "environment", return_value={}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--model", "model", "--prompts-file", str(prompts), "--num-samples", "3", "--batch-sizes", "2", "--output-dir", directory]), expect_status)
            run_dir = next(base.glob("local_*"))
            payload = {p.name: json.loads(p.read_text()) for p in run_dir.glob("*.json")}
        self.assertTrue(monitor.closed)
        self.assertTrue(monitor.stopped)
        return payload

    def test_run_saves_prompts_measurement_scope_and_raw_samples(self):
        payload = self.run_fixture(FakeEngine(), 0)
        self.assertEqual(payload["status.json"]["status"], "completed")
        self.assertEqual(payload["prompts.json"], ["prompt one", "prompt two", "prompt one"])
        self.assertIn("batch_2_power_samples.json", payload)
        result = payload["batch_2_result.json"]
        self.assertEqual(result["total_output_tokens"], 9)
        self.assertEqual(result["phase_status"], "not_requested")
        self.assertIsNone(result["energy"]["system_energy_j"])

    def test_failure_records_status_raw_samples_and_closes_monitor(self):
        class BrokenEngine(FakeEngine):
            def run_benchmark(self, *args):
                raise RuntimeError("inference failure")
        payload = self.run_fixture(BrokenEngine(), 1)
        self.assertEqual(payload["status.json"]["status"], "failed")
        self.assertIn("batch_2_power_samples.json", payload)
        self.assertNotIn("results.json", payload)

    def test_interrupt_records_terminal_state(self):
        class InterruptedEngine(FakeEngine):
            def run_benchmark(self, *args):
                raise KeyboardInterrupt
        payload = self.run_fixture(InterruptedEngine(), 130)
        self.assertEqual(payload["status.json"]["status"], "interrupted")

    def test_model_snapshot_requires_unambiguous_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for revision in ("aaa", "bbb"):
                snapshot = root / "snapshots" / revision
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                api.resolve_model_path(directory)
            (root / "refs").mkdir()
            (root / "refs" / "main").write_text("aaa")
            self.assertEqual(api.resolve_model_path(directory), str((root / "snapshots" / "aaa").resolve()))

    def test_dataset_failure_never_uses_builtin_prompts(self):
        from tokenpowerbench.data import loader
        with patch.object(loader, "_HF_AVAILABLE", False), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                loader.DatasetLoader().load("alpaca")


if __name__ == "__main__":
    unittest.main()

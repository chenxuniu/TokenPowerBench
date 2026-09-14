"""Public API/CLI behavior without importing inference or sensor dependencies."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tokenpowerbench import BenchmarkResult, EnergyMetrics, __version__, api, benchmark, check_environment, cli
from tokenpowerbench.engines import VLLMEngine


class StubMonitor:
    def __init__(self):
        self.capabilities = {"gpu": {"available": True, "devices": [{"index": 0}]}}
        self.samples = {"gpu": [{"timestamp_s": 0, "power_w": [100]}], "rapl": [], "ipmi": []}
        self.closed = False
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1

    def close(self):
        self.closed = True

    def compute_metrics(self, duration, tokens, responses, start_time=None, end_time=None):
        return EnergyMetrics(duration=duration, total_output_tokens=tokens, num_responses=responses,
                             start_time=start_time, end_time=end_time)


class StubEngine:
    def __init__(self, error=None):
        self.error = error
        self.closed = False
        self.requests = []
        self.resolved_config = {"requested": {}, "effective": {}, "engine_class": "StubEngine"}

    def setup_model(self, *args, **kwargs):
        self.setup_options = kwargs

    def run_inference(self, *args, **kwargs):
        return [object()]

    def run_benchmark(self, prompts, *args):
        if self.error is not None:
            raise self.error
        self.requests.append(list(prompts))
        return [SimpleNamespace(token_ids=[1, 2, 3]) for _ in prompts], 1.0, 3.0

    def estimate_tokens(self, outputs):
        return sum(len(output.token_ids) for output in outputs)

    def close(self):
        self.closed = True


class APITests(unittest.TestCase):
    def resources(self, engine=None, monitor=None):
        engine = engine or StubEngine()
        monitor = monitor or StubMonitor()
        for active_patch in (patch.object(api, "VLLMEngine", return_value=engine),
                             patch.object(api, "create_monitor", return_value=monitor),
                             patch.object(api, "environment", return_value={"packages": {"tokenpowerbench": "1.0.0"}})):
            active_patch.start()
            self.addCleanup(active_patch.stop)
        return engine, monitor

    def test_public_api_returns_saved_results_and_uses_prompt_count(self):
        engine, monitor = self.resources()
        prompts = ["one prompt", "two prompts"]
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark(model="test-model", prompts=prompts, output_dir=directory)
            self.assertIsInstance(result, BenchmarkResult)
            self.assertTrue(result.output_dir.is_absolute())
            self.assertEqual(json.loads((result.output_dir / "results.json").read_text()), result.results)
            self.assertEqual(json.loads(json.dumps(result.to_dict()))["output_dir"], str(result.output_dir))
            self.assertEqual(json.loads((result.output_dir / "prompts.json").read_text()), prompts)
            self.assertEqual(json.loads((result.output_dir / "config.json").read_text())["num_samples"], 2)
            self.assertEqual(json.loads((result.output_dir / "status.json").read_text())["status"], "completed")
        self.assertEqual(engine.requests, [prompts])
        self.assertEqual(result.results["batch_1"]["total_output_tokens"], 6)
        self.assertTrue(monitor.closed)
        self.assertTrue(engine.closed)

    def test_prompt_file_defaults_to_file_length_and_explicit_count_repeats(self):
        engine, monitor = self.resources()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "prompts.json"
            source.write_text('["first", "second"]')
            first = benchmark(model="test", prompts_file=source, output_dir=directory)
            second = benchmark(model="test", prompts_file=source, num_samples=3, output_dir=directory)
            self.assertNotEqual(first.output_dir, second.output_dir)
        self.assertEqual(engine.requests, [["first", "second"], ["first", "second", "first"]])

    def test_no_source_uses_alpaca_and_1000_requests(self):
        engine, monitor = self.resources()
        loader = Mock()
        loader.load.return_value = ["dataset prompt"]
        dataset_module = SimpleNamespace(DatasetLoader=Mock(return_value=loader))
        with patch.dict(sys.modules, {"tokenpowerbench.data": dataset_module}), tempfile.TemporaryDirectory() as directory:
            result = benchmark(model="test", output_dir=directory)
        loader.load.assert_called_once_with("alpaca", num_samples=1000, min_words=2, max_words=300)
        self.assertEqual(result.results["batch_1"]["num_responses"], 1000)

    def test_invalid_inputs_fail_before_hardware_or_artifacts(self):
        invalid = [
            {"model": ""}, {"model": Path("model")}, {"prompts": "single string"},
            {"prompts": []}, {"prompts": [" "]}, {"prompts": [1]},
            {"prompts": ["a"], "dataset": "alpaca"},
            {"prompts_file": "a.json", "dataset": "alpaca"},
            {"prompts": ["a"], "prompts_file": "a.json"},
            {"dataset": "unknown"}, {"batch_sizes": "1,2"},
            {"batch_sizes": []}, {"batch_sizes": [1, 1]}, {"batch_sizes": [0]},
            {"batch_sizes": [True]}, {"batch_sizes": [1.5]},
            {"phase_profiling": True, "batch_sizes": [2]}, {"phase_profiling": 1},
            {"num_samples": True}, {"num_samples": 0}, {"output_tokens": 0},
            {"seed": True}, {"seed": -1}, {"min_words": -1}, {"min_words": 10, "max_words": 2},
            {"max_model_len": 0}, {"tensor_parallel_size": False},
            {"temperature": float("nan")}, {"temperature": -1}, {"temperature": "0"},
            {"gpu_memory_utilization": float("inf")}, {"gpu_memory_utilization": 0},
            {"gpu_memory_utilization": 1.1}, {"gpu_memory_utilization": True},
            {"output_dir": ""}, {"monitor": "unknown"},
        ]
        with patch.object(api, "create_monitor") as create:
            for overrides in invalid:
                with self.subTest(overrides=overrides), self.assertRaises((ValueError, TypeError)):
                    benchmark(**{"model": "test", **overrides})
        create.assert_not_called()

    def test_failures_raise_original_exception_and_preserve_raw_samples(self):
        for error, status in [(RuntimeError("inference broke"), "failed"), (KeyboardInterrupt(), "interrupted")]:
            with self.subTest(status=status), contextlib.ExitStack() as stack:
                engine, monitor = StubEngine(error), StubMonitor()
                stack.enter_context(patch.object(api, "VLLMEngine", return_value=engine))
                stack.enter_context(patch.object(api, "create_monitor", return_value=monitor))
                stack.enter_context(patch.object(api, "environment", return_value={}))
                directory = stack.enter_context(tempfile.TemporaryDirectory())
                with self.assertRaises(type(error)) as caught:
                    benchmark(model="test", prompts=["prompt"], output_dir=directory)
                self.assertIs(caught.exception, error)
                run = next(Path(directory).glob("local_*"))
                self.assertEqual(json.loads((run / "status.json").read_text())["status"], status)
                self.assertEqual(json.loads((run / "batch_1_power_samples.json").read_text()), monitor.samples)
                self.assertFalse((run / "results.json").exists())
                self.assertTrue(monitor.closed)
                self.assertTrue(engine.closed)

    def test_api_and_cli_use_identical_measurement_pipeline(self):
        self.resources()
        with tempfile.TemporaryDirectory() as directory:
            prompt_file = Path(directory) / "prompts.json"
            prompt_file.write_text('["prompt"]')
            direct = benchmark(model="test", prompts_file=prompt_file, batch_sizes=(1, 2), output_dir=directory)
            with contextlib.redirect_stdout(io.StringIO()):
                status = cli.main(["--model", "test", "--prompts-file", str(prompt_file),
                                   "--batch-sizes", "1,2", "--output-dir", directory])
            self.assertEqual(status, 0)
            cli_dir = next(path for path in Path(directory).glob("local_*") if path != direct.output_dir)
            self.assertEqual(json.loads((cli_dir / "results.json").read_text()), direct.results)
            self.assertEqual(json.loads((cli_dir / "config.json").read_text()),
                             json.loads((direct.output_dir / "config.json").read_text()))

    def test_engine_core_shutdown_is_called_once_and_clears_model_reference(self):
        shutdown = Mock()
        engine = VLLMEngine()
        engine._llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)))
        engine.close()
        engine.close()
        shutdown.assert_called_once_with()
        self.assertIsNone(engine._llm)

    def test_v0_executor_shutdown_precedes_cuda_cache_release(self):
        order = []
        shutdown = Mock(side_effect=lambda: order.append("shutdown"))
        backend = SimpleNamespace(model_executor=SimpleNamespace(shutdown=shutdown))
        engine = VLLMEngine()
        engine._llm = SimpleNamespace(llm_engine=backend)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True,
            empty_cache=lambda: order.append("empty_cache")))
        with patch.dict(sys.modules, {"torch": torch}):
            engine.close()
            engine.close()
        self.assertEqual(order, ["shutdown", "empty_cache"])
        self.assertIsNone(backend.model_executor)

    def test_sequential_api_calls_close_previous_engine_before_loading_next(self):
        order = []
        class OrderedEngine(StubEngine):
            def setup_model(self, *args, **kwargs):
                order.append("setup")
            def close(self):
                order.append("close")
                super().close()
        with patch.object(api, "VLLMEngine", side_effect=OrderedEngine), \
             patch.object(api, "create_monitor", side_effect=lambda *args, **kwargs: StubMonitor()), \
             patch.object(api, "environment", return_value={}), tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                benchmark(model="test", prompts=["prompt"], output_dir=directory)
        self.assertEqual(order, ["setup", "close", "setup", "close"])

    def test_setup_failure_still_closes_engine_and_monitor(self):
        engine, sensor = self.resources()
        engine.setup_model = Mock(side_effect=RuntimeError("model setup failed"))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "model setup failed"):
            benchmark(model="test", prompts=["prompt"], output_dir=directory)
        self.assertTrue(engine.closed)
        self.assertTrue(sensor.closed)

    def test_shutdown_failure_marks_failed_and_still_closes_monitor(self):
        engine, sensor = self.resources()
        engine.close = Mock(side_effect=RuntimeError("engine shutdown failed"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "engine shutdown failed"):
                benchmark(model="test", prompts=["prompt"], output_dir=directory)
            run = next(Path(directory).glob("local_*"))
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "failed")
            self.assertTrue((run / "batch_1_power_samples.json").is_file())
        self.assertTrue(sensor.closed)

    def test_inference_failure_or_interrupt_remains_primary_when_cleanup_fails(self):
        for error, status in [(RuntimeError("inference failed"), "failed"), (KeyboardInterrupt(), "interrupted")]:
            with self.subTest(status=status), contextlib.ExitStack() as stack:
                engine, sensor = StubEngine(error), StubMonitor()
                engine.close = Mock(side_effect=RuntimeError("shutdown failed"))
                stack.enter_context(patch.object(api, "VLLMEngine", return_value=engine))
                stack.enter_context(patch.object(api, "create_monitor", return_value=sensor))
                stack.enter_context(patch.object(api, "environment", return_value={}))
                directory = stack.enter_context(tempfile.TemporaryDirectory())
                with self.assertRaises(type(error)) as caught:
                    benchmark(model="test", prompts=["prompt"], output_dir=directory)
                self.assertIs(caught.exception, error)
                run = next(Path(directory).glob("local_*"))
                recorded = json.loads((run / "status.json").read_text())
                self.assertEqual(recorded["status"], status)
                self.assertIn("shutdown failed", recorded["cleanup_errors"][0])
                self.assertTrue(sensor.closed)

    def test_interrupt_survives_sampling_stop_failure_and_keeps_raw_data(self):
        error = KeyboardInterrupt()
        engine, sensor = self.resources(engine=StubEngine(error))
        sensor.stop = Mock(side_effect=RuntimeError("stop failed"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt) as caught:
                benchmark(model="test", prompts=["prompt"], output_dir=directory)
            self.assertIs(caught.exception, error)
            run = next(Path(directory).glob("local_*"))
            recorded = json.loads((run / "status.json").read_text())
            self.assertEqual(recorded["status"], "interrupted")
            self.assertIn("monitor.stop", recorded["cleanup_errors"][0])
            self.assertEqual(json.loads((run / "batch_1_power_samples.json").read_text()), sensor.samples)
        self.assertTrue(engine.closed)
        self.assertTrue(sensor.closed)

    def test_phase_api_preserves_submission_boundary_and_exact_decode_count(self):
        class PhaseEngine(StubEngine):
            def run_profiled_benchmark(self, prompts, *args):
                self.requests.append(list(prompts))
                event = {"submitted_s": 1.0, "dispatch_completed_s": 1.1,
                         "prefill_start_s": 1.0, "first_token_s": 2.0,
                         "finished_s": 4.0, "output_tokens": 3}
                return [SimpleNamespace(token_ids=[1, 2, 3])], 1.0, 4.0, [event]
        engine, sensor = self.resources(engine=PhaseEngine())
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark(model="test", prompts=["prompt"], phase_profiling=True, output_dir=directory)
        self.assertEqual(len(engine.requests), 2)  # Warmup and measured request.
        batch = result.results["batch_1"]
        self.assertEqual(batch["phase_status"], "serial_host_boundaries")
        phase = batch["phases"][0]
        self.assertEqual(phase["prefill_proxy_energy"]["start_time"], 1.0)
        self.assertEqual(phase["prefill_proxy_energy"]["end_time"], 2.0)
        self.assertEqual(phase["decode_energy"]["total_output_tokens"], 2)

    def test_check_environment_forwards_explicit_devices_and_closes(self):
        sensor = StubMonitor()
        with patch.object(api, "create_monitor", return_value=sensor) as create, patch.object(api, "VLLMEngine") as engine:
            report = check_environment(monitor="gpu_only", device_indices=[0])
        create.assert_called_once_with("gpu_only", device_indices=[0], device_uuids=None)
        engine.assert_not_called()
        self.assertEqual(set(report), {"runtime", "sensors"})
        self.assertIn("is_root", report["runtime"])
        self.assertTrue(sensor.closed)

    def test_check_environment_closes_when_capability_read_fails(self):
        class BrokenCapabilities:
            close = Mock()
            @property
            def capabilities(self):
                raise RuntimeError("capabilities unavailable")
        sensor = BrokenCapabilities()
        with patch.object(api, "create_monitor", return_value=sensor), self.assertRaisesRegex(RuntimeError, "capabilities"):
            check_environment()
        sensor.close.assert_called_once()

    def test_installed_package_does_not_report_enclosing_unrelated_git_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / "pyproject.toml").write_text('[project]\nname = "unrelated-project"\n')
            installed_api = root / ".venv/lib/site-packages/tokenpowerbench/api.py"
            with patch.object(api, "__file__", str(installed_api)), patch.object(api.platform, "platform", return_value="test-platform"), patch.object(api.subprocess, "check_output") as git:
                data = api.environment()
        git.assert_not_called()
        self.assertIsNone(data["git_commit"])
        self.assertIsNone(data["git_status"])
        self.assertEqual(data["packages"]["tokenpowerbench"], __version__)

    def test_import_help_and_version_do_not_import_optional_dependencies(self):
        code = '''import sys
import tokenpowerbench
from tokenpowerbench import benchmark, check_environment, BenchmarkResult, create_monitor, EnergyMetrics
from tokenpowerbench.cli import main
for args in (["--help"], ["--version"]):
    try:
        main(args)
    except SystemExit as result:
        assert result.code == 0
assert not {"torch", "vllm", "datasets", "transformers", "pynvml"}.intersection(sys.modules)
'''
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tokenpowerbench 1.0.0", result.stdout)
        module = subprocess.run([sys.executable, "-m", "tokenpowerbench", "--version"], text=True, capture_output=True)
        self.assertEqual(module.returncode, 0, module.stderr)
        self.assertEqual(module.stdout.strip(), "tokenpowerbench 1.0.0")


if __name__ == "__main__":
    unittest.main()

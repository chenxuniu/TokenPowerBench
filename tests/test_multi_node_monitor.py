"""Multi-node CLI artifact, lifecycle, and energy-scope checks without hardware."""

from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tokenpowerbench.energy import EnergyMetrics


def load_runner():
    distributed = ModuleType("tokenpowerbench.distributed")
    distributed.VLLMDistributedEngine = Mock()
    distributed.RayClusterConfig = SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(ray_init_address="test:6379")))
    spec = importlib.util.spec_from_file_location(
        "multi_node_monitor_fixture", Path(__file__).resolve().parents[1] / "run_multi_node.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"tokenpowerbench.distributed": distributed}):
        spec.loader.exec_module(module)
    return module


class MultiNodeMonitorCompatibilityTests(unittest.TestCase):
    @contextmanager
    def fixture(self, monitor_mode="gpu_only", extra_args=()):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            inputs = root / "prompts.json"
            inputs.write_text('["first prompt", "second prompt"]')
            argv = ["--models", "model", "--model-dir", str(root / "worker-only-models"),
                    "--prompts-file", str(inputs), "--tensor-parallel", "1", "--pipeline-parallel", "1",
                    "--batch-sizes", "1", "--monitor", monitor_mode, "--output-dir", str(root / "results"), *extra_args]
            args = runner.parse_args(argv)
            events = []
            engine = Mock()
            engine.prepare.side_effect = lambda: events.append("prepare") or {"workers": [{"id": 0}]}
            result = {"performance_metrics": {"total_tokens": 100, "total_prompts": 2},
                      "measurement_window": {"start_s": 10.0, "end_s": 12.0,
                                             "clock": "time.perf_counter", "scope": "driver_dispatch_and_collection"},
                      "results": []}
            engine.run_benchmark.side_effect = lambda prompts: events.append("infer") or copy.deepcopy(result)
            engine.close.side_effect = lambda: events.append("engine.close")
            monitor = Mock()
            monitor.samples = {"gpu": [{"timestamp_s": 10.0, "power_w": [100.0]}], "ipmi": [], "rapl": []}
            monitor.capabilities = {"gpu": {"available": True}}
            monitor.start.side_effect = lambda: events.append("start")
            monitor.stop.side_effect = lambda: events.append("stop")
            monitor.close.side_effect = lambda: events.append("monitor.close")
            monitor.compute_metrics.return_value = EnergyMetrics(duration=2.0, total_output_tokens=0, num_responses=0,
                                                                 gpu_energy_j=200.0, energy_scope="selected_gpus")
            stack.enter_context(patch.object(runner, "parse_args", return_value=args))
            stack.enter_context(patch.object(runner, "VLLMDistributedEngine", return_value=engine))
            factory = stack.enter_context(patch.object(runner, "create_monitor", return_value=monitor))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            yield SimpleNamespace(runner=runner, engine=engine, monitor=monitor, factory=factory, args=args,
                                  output=root / "results", events=events, result=result)

    @staticmethod
    def suite(fixture):
        return next(fixture.output.glob("multi_*"))

    @staticmethod
    def configuration(fixture):
        return next(MultiNodeMonitorCompatibilityTests.suite(fixture).glob("config_*"))

    def test_success_excludes_startup_and_never_attributes_driver_energy_to_cluster(self):
        with self.fixture() as f:
            self.assertEqual(f.runner.run(), 0)
            self.assertEqual(f.events, ["prepare", "start", "infer", "stop", "engine.close", "monitor.close"])
            f.monitor.compute_metrics.assert_called_once_with(2.0, 0, 0, start_time=10.0, end_time=12.0)
            folder = self.configuration(f)
            result = json.loads((folder / "result.json").read_text())
            self.assertEqual(result["energy_metrics"]["energy_scope"], "unmeasured_cluster")
            self.assertIsNone(result["energy_metrics"]["cluster_energy_j"])
            self.assertIsNone(result["energy_metrics"]["total_mj_per_token"])
            driver = result["driver_diagnostics"]
            self.assertEqual(driver["monitoring_scope"], "local_driver_host")
            self.assertEqual(driver["energy"]["gpu_energy_j"], 200.0)
            self.assertNotIn("gpu_mj_per_token", driver["energy"])
            self.assertEqual(json.loads((folder / "driver_power_samples.json").read_text()), f.monitor.samples)
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "completed")
            self.assertEqual(json.loads((self.suite(f) / "prompts.json").read_text())["prompts_file"], ["first prompt", "second prompt"])

    def test_cpu_only_driver_does_not_probe_gpu_and_worker_model_need_not_exist_locally(self):
        with self.fixture(monitor_mode="none") as f:
            self.assertFalse(f.args.model_dir.exists())
            self.assertEqual(f.runner.run(), 0)
            f.factory.assert_not_called()
            self.assertEqual(f.events, ["prepare", "infer", "engine.close"])
            result = json.loads((self.configuration(f) / "result.json").read_text())
            self.assertEqual(result["driver_diagnostics"]["monitoring_scope"], "not_collected")
            self.assertIsNone(result["energy_metrics"]["total_energy_j"])

    def test_failure_preserves_raw_samples_and_returns_failure_status(self):
        with self.fixture() as f:
            f.engine.run_benchmark.side_effect = RuntimeError("worker lost")
            self.assertEqual(f.runner.run(), 1)
            folder = self.configuration(f)
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "failed")
            self.assertTrue((folder / "driver_power_samples.json").exists())
            self.assertIn("worker lost", (self.suite(f) / "failures.json").read_text())
            f.engine.close.assert_called_once()
            f.monitor.close.assert_called_once()

    def test_interrupt_survives_monitor_and_engine_cleanup_failures(self):
        with self.fixture() as f:
            f.engine.run_benchmark.side_effect = KeyboardInterrupt()
            f.monitor.stop.side_effect = RuntimeError("stop failed")
            f.engine.close.side_effect = RuntimeError("close failed")
            self.assertEqual(f.runner.run(), 130)
            folder = self.configuration(f)
            status = json.loads((folder / "status.json").read_text())
            self.assertEqual(status["status"], "interrupted")
            self.assertEqual(len(status["cleanup_errors"]), 2)
            self.assertTrue((folder / "driver_power_samples.json").exists())
            f.monitor.close.assert_called_once()

    def test_prepare_failure_never_starts_measurement_and_closes_resources(self):
        with self.fixture() as f:
            f.engine.prepare.side_effect = RuntimeError("placement timeout")
            self.assertEqual(f.runner.run(), 1)
            f.monitor.start.assert_not_called()
            f.engine.close.assert_called_once()
            f.monitor.close.assert_called_once()

    def test_monitor_probe_failure_closes_created_engine(self):
        with self.fixture() as f:
            f.factory.side_effect = RuntimeError("GPU inaccessible")
            self.assertEqual(f.runner.run(), 1)
            f.engine.close.assert_called_once()
            f.engine.prepare.assert_not_called()

    def test_partial_failure_preserves_success_and_reports_nonzero_exit(self):
        with self.fixture(extra_args=("--batch-sizes", "1,2")) as f:
            f.engine.run_benchmark.side_effect = [copy.deepcopy(f.result), RuntimeError("second replica failed")]
            self.assertEqual(f.runner.run(), 1)
            root = self.suite(f)
            self.assertEqual(len(json.loads((root / "results.json").read_text())), 1)
            self.assertEqual(json.loads((root / "status.json").read_text())["status"], "partial_failure")
            self.assertEqual(f.engine.close.call_count, 2)

    def test_none_or_incomplete_output_is_failure(self):
        for result in (None, {"performance_metrics": {"total_tokens": 10, "total_prompts": 1}}):
            with self.subTest(result=result), self.fixture() as f:
                f.engine.run_benchmark.side_effect = None
                f.engine.run_benchmark.return_value = result
                self.assertEqual(f.runner.run(), 1)
                f.monitor.compute_metrics.assert_not_called()

    def test_explicit_driver_devices_and_new_artifact_directories(self):
        with self.fixture(extra_args=("--driver-device-indices", "2,0")) as f:
            self.assertEqual(f.runner.run(), 0)
            self.assertEqual(f.runner.run(), 0)
            self.assertEqual(len(list(f.output.glob("multi_*"))), 2)
            f.factory.assert_called_with("gpu_only", device_indices=[2, 0])

    def test_invalid_arguments_fail_before_connecting(self):
        runner = load_runner()
        cases = [("--tensor-parallel", "0"), ("--batch-sizes", "1,1"), ("--models", "model,"),
                 ("--concurrency", "-1"), ("--temperature", "nan"), ("--top-p", "0"),
                 ("--seed", "-1"), ("--startup-timeout-s", "inf"), ("--ray-head-port", "70000"),
                 ("--driver-device-indices", "0"), ("--gpu-memory-utilization", "nan")]
        for values in cases:
            with self.subTest(values=values), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                runner.parse_args(["--models", "model", *values])
            self.assertEqual(exc.exception.code, 2)
        runner.RayClusterConfig.resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()

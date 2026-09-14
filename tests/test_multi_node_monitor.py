"""Check shared-monitor compatibility without starting Ray or CUDA workers."""

from contextlib import ExitStack, contextmanager, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from tokenpowerbench.energy import EnergyMetrics


def load_runner():
    distributed = ModuleType("tokenpowerbench.distributed")
    distributed.VLLMDistributedEngine = Mock()
    cluster = ModuleType("tokenpowerbench.distributed.ray_cluster")
    cluster.RayClusterConfig = Mock(return_value=SimpleNamespace(ray_init_address="test"))
    path = Path(__file__).resolve().parents[1] / "run_multi_node.py"
    spec = importlib.util.spec_from_file_location("multi_node_monitor_fixture", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "tokenpowerbench.distributed": distributed,
        "tokenpowerbench.distributed.ray_cluster": cluster,
    }):
        spec.loader.exec_module(module)
    return module


class MultiNodeMonitorCompatibilityTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        runner = load_runner()
        engine = Mock()
        engine.run_benchmark.return_value = {
            "performance_metrics": {"total_tokens": 100, "total_prompts": 2},
        }
        monitor = Mock()
        monitor.compute_metrics.return_value = EnergyMetrics(
            duration=10.0, total_output_tokens=100, num_responses=2,
            gpu_energy_j=1000.0, energy_scope="selected_gpus",
        )
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root / "model").mkdir()
            args = SimpleNamespace(
                models="model", model_dir=directory, datasets="alpaca", num_samples=2,
                min_words=1, max_words=100, tensor_parallel="1", pipeline_parallel="1",
                concurrency="1", batch_sizes="1", max_tokens=50, temperature=0.0,
                top_p=1.0, ray_head_address="test", ray_head_port=6379,
                monitor="gpu_only", output_dir=str(root / "results"), verbose=False,
            )
            stack.enter_context(patch.object(runner, "parse_args", return_value=args))
            stack.enter_context(patch.object(runner, "DatasetLoader", return_value=SimpleNamespace(
                load=lambda *args, **kwargs: ["first prompt", "second prompt"],
            )))
            stack.enter_context(patch.object(runner, "VLLMDistributedEngine", return_value=engine))
            stack.enter_context(patch.object(runner, "create_monitor", return_value=monitor))
            stack.enter_context(patch.object(runner.time, "perf_counter", side_effect=[10.0, 20.0]))
            sleep = stack.enter_context(patch.object(runner.time, "sleep"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield runner, engine, monitor, root / "results", sleep

    def test_success_uses_actual_monotonic_window_and_labels_driver_scope(self):
        with self.fixture() as (runner, engine, monitor, output, sleep):
            runner.run()
            monitor.compute_metrics.assert_called_once_with(
                10.0, 100, 2, start_time=10.0, end_time=20.0,
            )
            self.assertEqual(monitor.method_calls, [
                call.start(), call.stop(),
                call.compute_metrics(10.0, 100, 2, start_time=10.0, end_time=20.0),
                call.close(),
            ])
            sleep.assert_not_called()
            result = json.loads(next(output.glob("model_*.json")).read_text())
            energy = result["energy_metrics"]
            self.assertEqual(energy["monitoring_scope"], "local_driver_host")
            self.assertEqual(energy["energy_scope"], "selected_gpus")
            self.assertEqual(energy["token_denominator_scope"], "all_distributed_responses")
            self.assertEqual((energy["start_s"], energy["end_s"]), (10.0, 20.0))

    def test_reported_benchmark_failure_still_releases_monitor(self):
        with self.fixture() as (runner, engine, monitor, output, sleep):
            engine.run_benchmark.return_value = None
            runner.run()
            self.assertEqual(monitor.method_calls, [call.start(), call.stop(), call.close()])
            self.assertEqual(list(output.glob("model_*.json")), [])

    def test_exception_or_interrupt_stops_and_closes_monitor(self):
        for error in (RuntimeError("inference failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), self.fixture() as fixture:
                runner, engine, monitor, output, sleep = fixture
                engine.run_benchmark.side_effect = error
                with self.assertRaises(type(error)):
                    runner.run()
                self.assertEqual(monitor.method_calls, [call.start(), call.stop(), call.close()])

    def test_start_or_stop_failure_still_closes_monitor(self):
        for method in ("start", "stop"):
            with self.subTest(method=method), self.fixture() as fixture:
                runner, engine, monitor, output, sleep = fixture
                getattr(monitor, method).side_effect = RuntimeError("monitor failed")
                with self.assertRaisesRegex(RuntimeError, "monitor failed"):
                    runner.run()
                monitor.close.assert_called_once_with()
                monitor.compute_metrics.assert_not_called()


if __name__ == "__main__":
    unittest.main()

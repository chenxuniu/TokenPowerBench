"""Source-checkout adapters use the strict shared implementations."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from MultipleNode import dataset_loader, power_monitor, vllm_engine
from tokenpowerbench.energy import EnergyMetrics


class MultiNodeAdapterTests(unittest.TestCase):
    def test_imports_do_not_load_inference_datasets_or_sensor_dependencies(self):
        code = """
import builtins
original = builtins.__import__
blocked = {'ray', 'vllm', 'torch', 'numpy', 'datasets', 'pynvml'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise RuntimeError('unexpected dependency import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import MultipleNode.dataset_loader
import MultipleNode.power_monitor
import MultipleNode.vllm_engine
"""
        result = subprocess.run([sys.executable, "-S", "-c", code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_engine_constructor_resolves_cluster_and_retains_shared_lifecycle(self):
        config = {"model_path": "worker-model"}
        cluster = object()
        timer = Mock()
        with patch.object(vllm_engine.RayClusterConfig, "resolve", return_value=cluster), \
                patch.object(vllm_engine._Engine, "__init__", return_value=None) as initialize:
            engine = vllm_engine.VLLMDistributedEngine(config, timer)
        initialize.assert_called_once_with(cluster, config)
        self.assertIs(type(engine).run_benchmark, vllm_engine._Engine.run_benchmark)
        self.assertIs(type(engine).prepare, vllm_engine._Engine.prepare)
        self.assertIs(type(engine).close, vllm_engine._Engine.close)
        timer.record.assert_not_called()

    def test_engine_does_not_replace_worker_errors_with_none(self):
        error = RuntimeError("worker failed")
        with patch.object(vllm_engine._Engine, "__init__", return_value=None), \
                patch.object(vllm_engine._Engine, "run_benchmark", side_effect=error):
            engine = vllm_engine.VLLMDistributedEngine({})
            with self.assertRaises(RuntimeError) as raised:
                engine.run_benchmark(["prompt"])
        self.assertIs(raised.exception, error)

    def test_predictor_accepts_timer_without_creating_false_first_token_events(self):
        timer = Mock()
        params = {"max_tokens": 2}
        with patch.object(vllm_engine._Predictor, "__init__", return_value=None) as initialize:
            predictor = vllm_engine.VLLMPredictor("model", 2, 3, params, timer, True,
                                                 model_kwargs={"dtype": "auto"})
        initialize.assert_called_once_with("model", 2, 3, params, verbose=True,
                                           model_kwargs={"dtype": "auto"})
        self.assertIs(type(predictor).__call__, vllm_engine._Predictor.__call__)
        timer.record.assert_not_called()

    def test_dataset_aliases_forward_seed_cache_and_word_limits(self):
        shared = Mock()
        shared.load.return_value = ["actual dataset prompt"]
        with patch("tokenpowerbench.data.DatasetLoader", return_value=shared) as factory:
            loader = dataset_loader.DatasetLoader(cache_dir="cache", seed=19)
            factory.assert_not_called()
            self.assertEqual(loader.load_dataset("alpaca", 7, 2, 80), ["actual dataset prompt"])
            shared.load.assert_called_once_with("alpaca", num_samples=7, min_words=2, max_words=80)
            loader.load("dolly", 9, 3, 50)
            shared.load.assert_called_with("dolly", num_samples=9, min_words=3, max_words=50)
            factory.assert_called_once_with(cache_dir="cache", seed=19)
            self.assertEqual(loader.get_dataset_info(), shared.supported_datasets.return_value)

    def test_dataset_failures_and_unknown_names_do_not_return_synthetic_prompts(self):
        from tokenpowerbench.data import loader as shared
        with patch.object(shared, "_HF_AVAILABLE", False), redirect_stdout(io.StringIO()):
            loader = dataset_loader.DatasetLoader()
            for name in ("alpaca", "dolly", "longbench", "humaneval"):
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    loader.load_dataset(name)
            with self.assertRaises(ValueError):
                loader.load_dataset("unknown")

    def test_dataset_sampling_is_reproducible_without_global_random_mutation(self):
        from tokenpowerbench.data import loader as shared
        data = {"train": [{"instruction": f"Actual instruction number {i}"} for i in range(20)]}
        previous_state = random.getstate()
        with patch.object(shared, "_HF_AVAILABLE", True), \
                patch.object(shared, "hf_load_dataset", return_value=data, create=True), \
                redirect_stdout(io.StringIO()):
            first = dataset_loader.DatasetLoader(seed=17).load_dataset("alpaca", 5, 1, 100)
            second = dataset_loader.DatasetLoader(seed=17).load_dataset("alpaca", 5, 1, 100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(random.getstate(), previous_state)

    def test_power_monitor_reports_only_local_diagnostics_without_global_normalization(self):
        monitor = Mock()
        monitor.capabilities = {"ipmi": {"available": True}}
        monitor.samples = {"gpu": [], "ipmi": []}
        monitor.compute_metrics.return_value = EnergyMetrics(
            duration=2.0, start_time=10.0, end_time=12.0, gpu_energy_j=200,
            system_energy_j=600, system_avg_power_w=300, energy_scope="whole_node")
        with patch.object(power_monitor, "create_monitor", return_value=monitor) as factory:
            adapter = power_monitor.PowerMonitor("auto", device_indices=[2])
            adapter.start_monitoring()
            adapter.stop_monitoring()
            result = adapter.calculate_metrics(2.0, 1000, 20, start_time=10.0, end_time=12.0)
            adapter.close()
        factory.assert_called_once_with("auto", device_indices=[2], device_uuids=None)
        monitor.compute_metrics.assert_called_once_with(2.0, 0, 0, start_time=10.0, end_time=12.0)
        for method in (monitor.start, monitor.stop, monitor.close):
            method.assert_called_once_with()
        self.assertEqual(result["energy_scope"], "unmeasured_cluster")
        self.assertEqual(result["monitoring_scope"], "local_driver_host")
        for key in ("cluster_energy_j", "cluster_energy_per_token_j", "total_energy_j",
                    "total_energy", "gpu_energy", "cpu_energy", "energy_per_token",
                    "gpu_energy_per_token", "total_energy_per_response", "gpu_mj_per_token"):
            self.assertIsNone(result[key], key)
        local = result["driver_diagnostics"]["energy"]
        self.assertEqual(local["system_energy_j"], 600)
        self.assertEqual(local["gpu_energy_j"], 200)
        self.assertNotIn("total_output_tokens", local)
        self.assertNotIn("energy_per_token_j", local)
        self.assertEqual(result["power_samples"], monitor.samples)

    def test_power_requires_explicit_driver_window_and_preserves_unavailable_sensors(self):
        monitor = Mock()
        monitor.compute_metrics.return_value = EnergyMetrics(duration=2, energy_scope="selected_gpus")
        with patch.object(power_monitor, "create_monitor", return_value=monitor):
            adapter = power_monitor.PowerMonitor()
        for endpoints in ({}, {"start_time": 10}, {"end_time": 12}):
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                adapter.calculate_metrics(2, 100, **endpoints)
        monitor.compute_metrics.assert_not_called()
        result = adapter.calculate_metrics(2, 100, start_time=10, end_time=12)
        self.assertIsNone(result["driver_diagnostics"]["energy"]["system_energy_j"])


if __name__ == "__main__":
    unittest.main()

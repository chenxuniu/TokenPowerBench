"""Hardware-free regression tests for monitoring permissions and energy math."""

import builtins
import json
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from tokenpowerbench.energy import EnergyMetrics, GPUEnergyMonitor, create_monitor
from tokenpowerbench.energy import full_node_monitor as full
from tokenpowerbench.energy.gpu_monitor import integrate_power


class FakeNVML:
    class NVMLError_NotSupported(Exception):
        pass

    def __init__(self):
        self.nvmlInit = Mock()
        self.nvmlShutdown = Mock()
        self.nvmlDeviceGetCount = Mock(return_value=4)
        self.nvmlDeviceGetHandleByIndex = Mock(side_effect=lambda index: index)
        self.nvmlDeviceGetHandleByUUID = Mock(side_effect=lambda uuid: int(uuid.split("-")[-1]))
        self.nvmlDeviceGetIndex = Mock(side_effect=lambda handle: handle)
        self.nvmlDeviceGetUUID = Mock(side_effect=lambda handle: f"GPU-{handle}")
        self.nvmlDeviceGetName = Mock(side_effect=lambda handle: f"Test GPU {handle}".encode())
        self.nvmlDeviceGetMigMode = Mock(return_value=(0, 0))
        self.nvmlDeviceGetPowerUsage = Mock(side_effect=lambda handle: (handle + 1) * 100_000)


class IntegrationTests(unittest.TestCase):
    def test_irregular_timestamp_trapezoids(self):
        result = integrate_power([(0, 100), (0.2, 200), (1, 100), (2, 300)], 0, 2)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["energy_j"], 350)
        self.assertAlmostEqual(result["average_power_w"], 175)

    def test_clipped_window_interpolates_boundaries_without_idle_trim(self):
        result = integrate_power([(0, 100), (0.5, 150), (1, 200), (2, 300)], 0.25, 1.75)
        self.assertAlmostEqual(result["energy_j"], 300)
        self.assertEqual(result["sample_count"], 2)
        # Huge idle-tail values cannot alter integration inside exact endpoints.
        result = integrate_power([(-10, 9000), (0, 100), (1, 100), (2, 100), (10, 9000)], 0, 2)
        self.assertEqual(result["energy_j"], 200)

    def test_missing_coverage_is_unavailable(self):
        for start, end in [(-1, 1), (0, 3)]:
            self.assertIsNone(integrate_power([(0, 100), (1, 100), (2, 100)], start, end)["energy_j"])

    def test_missing_and_invalid_values_are_not_zero(self):
        for missing in (None, float("nan"), float("inf"), -1):
            result = integrate_power([(0, 100), (1, missing), (2, 100)], 0, 2)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["energy_j"])
        result = integrate_power([(-1, None), (0, 100), (1, 100), (2, None)], 0, 1)
        self.assertEqual(result["energy_j"], 100)

    def test_zero_is_a_valid_reading(self):
        self.assertEqual(integrate_power([(0, 0), (1, 0)], 0, 1)["energy_j"], 0)

    def test_sparse_and_unsorted_samples_are_invalid(self):
        self.assertFalse(integrate_power([(0, 100), (2, 100)], 0.1, 1.9)["valid"])
        self.assertFalse(integrate_power([(0, 100), (0, 100), (2, 100)], 0, 2)["valid"])
        with self.assertRaises(ValueError):
            integrate_power([(0, 100), (2, 100)], 1, 1)

    def test_counter_interpolation_and_wraparound(self):
        result = full.integrate_energy_counter([(0, 0), (1, 100_000_000), (2, 300_000_000),
                                                (3, 400_000_000)], 0.5, 2.5)
        self.assertEqual(result["energy_j"], 300)
        result = full.integrate_energy_counter([(0, 900_000), (1, 100_000), (2, 300_000)],
                                               0, 2, max_energy_uj=1_000_000)
        self.assertAlmostEqual(result["energy_j"], 0.4)
        result = full.integrate_energy_counter([(0, 900_000), (1, 100_000), (2, 300_000)], 0, 2)
        self.assertIsNone(result["energy_j"])

    def test_prefill_and_decode_partition_nonconstant_energy(self):
        readings = [(0, 20), (0.2, 150), (1.6, 40), (2, 300), (3, 70)]
        whole = integrate_power(readings, 0.1, 2.7, min_samples=0)
        prefill = integrate_power(readings, 0.1, 1.1, min_samples=0)
        decode = integrate_power(readings, 1.1, 2.7, min_samples=0)
        self.assertAlmostEqual(whole["energy_j"], prefill["energy_j"] + decode["energy_j"])

    def test_missing_rapl_counter_invalidates_window(self):
        result = full.integrate_energy_counter([(0, 0), (1, None), (2, 300_000_000)], 0, 2)
        self.assertIsNone(result["energy_j"])


class NVMLTestCase(unittest.TestCase):
    def setUp(self):
        self.nvml = FakeNVML()
        self.module_patch = patch.dict(sys.modules, {"pynvml": self.nvml})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def monitor(self, **kwargs):
        monitor = GPUEnergyMonitor(**kwargs)
        self.addCleanup(monitor.close)
        return monitor


class MonitorTests(NVMLTestCase):

    def test_selected_physical_gpu_only(self):
        monitor = self.monitor(device_indices=[2])
        monitor._samples["gpu"] = [{"timestamp_s": t, "power_w": [300]} for t in (0, 1, 2)]
        metrics = monitor.compute_metrics(2, 100, 1, start_time=0, end_time=2)
        self.assertEqual(metrics.gpu_energy_j, 600)
        self.assertEqual(metrics.per_gpu_energy_j, {2: 600})
        self.assertEqual(metrics.energy_scope, "selected_gpus")
        self.assertIsNone(metrics.system_energy_j)
        self.assertIsNone(metrics.total_energy_j)
        self.assertEqual(metrics.components_energy_j, 600)
        self.assertEqual(monitor.capabilities["gpu"]["devices"][0]["uuid"], "GPU-2")
        self.assertEqual([call.args[0] for call in self.nvml.nvmlDeviceGetPowerUsage.call_args_list], [2])
        json.dumps({"metrics": asdict(metrics), "capabilities": monitor.capabilities,
                    "samples": monitor.samples}, allow_nan=False)

    def test_default_selection_uses_torch_uuid_mapping(self):
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            device_count=lambda: 1,
            get_device_properties=lambda index: types.SimpleNamespace(uuid="GPU-3")))
        with patch.dict(sys.modules, {"torch": torch}):
            monitor = self.monitor()
        self.assertEqual(monitor.capabilities["gpu"]["devices"][0]["index"], 3)
        self.nvml.nvmlDeviceGetCount.assert_not_called()
        self.nvml.nvmlDeviceGetHandleByIndex.assert_not_called()

    def test_missing_uuid_does_not_monitor_all_host_gpus(self):
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            device_count=lambda: 1, get_device_properties=lambda index: object()))
        with patch.dict(sys.modules, {"torch": torch}), self.assertRaisesRegex(RuntimeError, "UUID"):
            GPUEnergyMonitor()
        self.nvml.nvmlShutdown.assert_called_once()

    def test_mig_rejected_and_init_failure_cleans_up(self):
        self.nvml.nvmlDeviceGetMigMode.return_value = (1, 1)
        with self.assertRaisesRegex(ValueError, "MIG"):
            GPUEnergyMonitor(device_indices=[0])
        self.nvml.nvmlShutdown.assert_called_once()

    def test_unsupported_mig_query_is_normal_on_older_gpus(self):
        self.nvml.nvmlDeviceGetMigMode.side_effect = FakeNVML.NVMLError_NotSupported()
        monitor = self.monitor(device_indices=[0])
        self.assertTrue(monitor.capabilities["gpu"]["available"])

    def test_bad_selection_does_not_silently_widen_scope(self):
        for selection in ([], [-1], [0, 0], [False]):
            with self.assertRaises(ValueError):
                GPUEnergyMonitor(device_indices=selection)
        self.nvml.nvmlInit.assert_not_called()
        with self.assertRaisesRegex(ValueError, "out of range"):
            GPUEnergyMonitor(device_indices=[4])
        self.nvml.nvmlShutdown.assert_called_once()

    def test_gpu_only_does_not_probe_privileged_sensors(self):
        with patch.object(full, "_RaplReader", side_effect=AssertionError("unexpected RAPL probe")), \
                patch.object(full, "_probe_ipmi_power", side_effect=AssertionError("unexpected IPMI probe")):
            monitor = create_monitor("gpu_only", device_indices=[0])
        self.addCleanup(monitor.close)
        self.assertFalse(monitor.capabilities["cpu"]["available"])
        self.assertIn("disabled", monitor.capabilities["cpu"]["reason"])

    def test_short_gpu_window_does_not_fabricate_phase_energy(self):
        monitor = self.monitor(device_indices=[0])
        monitor._samples["gpu"] = [{"timestamp_s": t, "power_w": [100]} for t in (0, 0.1, 0.2, 0.3)]
        metrics = monitor.compute_metrics(0.2, 10, 1, start_time=0, end_time=0.2)
        self.assertIsNone(metrics.gpu_energy_j)
        self.assertTrue(any("1-second NVML" in warning for warning in metrics.warnings))
        self.assertEqual(monitor.capabilities["gpu"]["minimum_window_s"], 1.0)

    def test_sensor_errors_record_null_and_invalidate_aggregate(self):
        monitor = self.monitor(device_indices=[0, 1])
        self.nvml.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("sensor permission revoked")
        monitor._sample_gpu()
        self.assertEqual(monitor.samples["gpu"][0]["power_w"], [None, None])
        self.assertIn("gpu:0", monitor.capabilities["gpu"]["sampling_errors"])
        monitor._samples["gpu"] = [{"timestamp_s": 0, "power_w": [100, 200]},
                                   {"timestamp_s": 1, "power_w": [100, None]},
                                   {"timestamp_s": 2, "power_w": [100, 200]}]
        metrics = monitor.compute_metrics(2, 100, 1, start_time=0, end_time=2)
        self.assertIsNone(metrics.gpu_energy_j)
        self.assertEqual(metrics.per_gpu_energy_j[0], 200)
        self.assertIsNone(metrics.per_gpu_energy_j[1])

    def test_stop_restart_and_close_release_nvml_once(self):
        monitor = self.monitor(device_indices=[0], interval_s=60)
        monitor.start()
        self.assertEqual(len(monitor.samples["gpu"]), 1)
        monitor.stop()
        self.assertEqual(len(monitor.samples["gpu"]), 2)
        self.assertFalse(any(thread.is_alive() for thread in monitor._threads))
        self.nvml.nvmlShutdown.assert_not_called()
        monitor.start()
        self.assertEqual(len(monitor.samples["gpu"]), 1)
        monitor.close()
        monitor.close()
        self.nvml.nvmlShutdown.assert_called_once()
        self.assertFalse(any(thread.is_alive() for thread in monitor._threads))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            monitor.start()

    def test_missing_metrics_remain_null_instead_of_zero(self):
        metrics = EnergyMetrics(total_output_tokens=10)
        self.assertIsNone(metrics.gpu_mj_per_token)
        self.assertIsNone(metrics.total_mj_per_token)
        self.assertIsNone(metrics.components_energy_j)
        self.assertIn("unavailable", metrics.summary())

    def test_deadline_cadence_does_not_add_sensor_latency(self):
        monitor = self.monitor(device_indices=[0])
        now = [0.0]
        waits = []

        def wait(delay):
            waits.append(delay)
            now[0] += delay
            return len(waits) == 4

        def acquire():
            now[0] += 0.3

        monitor._stop_event = types.SimpleNamespace(wait=wait)
        with patch("tokenpowerbench.energy.gpu_monitor.time.perf_counter", side_effect=lambda: now[0]):
            monitor._sample_loop(acquire, 1.0)
        self.assertAlmostEqual(waits[0], 1.0)
        self.assertAlmostEqual(waits[1], 0.7)
        self.assertAlmostEqual(waits[2], 0.7)


class FullNodeTests(NVMLTestCase):
    """Optional-sensor permission, accounting, and capability regressions."""

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root_patch = patch.object(full, "_RAPL_ROOT", str(self.root))
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def domain(self, path, name, energy=1_000_000):
        directory = self.root / path
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "name").write_text(name)
        (directory / "energy_uj").write_text(str(energy))
        (directory / "max_energy_range_uj").write_text("1000000000")
        return directory

    def full_monitor(self, strict=False):
        monitor = full.FullNodeEnergyMonitor(device_indices=[0], strict=strict)
        self.addCleanup(monitor.close)
        return monitor

    def test_rapl_discards_core_uncore_and_psys_overlap(self):
        self.domain("intel-rapl:0", "package-0")
        self.domain("intel-rapl:0/intel-rapl:0:0", "core")
        self.domain("intel-rapl:0/intel-rapl:0:1", "uncore")
        dram = self.domain("intel-rapl:0/intel-rapl:0:2", "dram")
        self.domain("intel-rapl:1", "psys")
        # A second sysfs alias to the same domain must not duplicate energy.
        (self.root / "intel-rapl:0:2").symlink_to(dram)
        reader = full._RaplReader()
        self.assertEqual(len(reader.domains), 2)
        self.assertEqual(sorted(reader.kinds.values()), ["cpu", "dram"])

    def test_permission_denied_is_detected_by_actual_read(self):
        denied = self.domain("intel-rapl:0", "package-0") / "energy_uj"
        original_open = builtins.open

        def read_with_denial(path, *args, **kwargs):
            if str(path) == str(denied):
                raise PermissionError("permission denied by test")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=read_with_denial), \
                patch.object(full, "_probe_ipmi_power", return_value=(None, "IPMI permission denied")):
            monitor = create_monitor("auto", device_indices=[0])
        self.addCleanup(monitor.close)
        self.assertFalse(monitor.capabilities["cpu"]["available"])
        self.assertIn("permission denied", monitor.capabilities["cpu"]["reason"])
        self.assertFalse(monitor.capabilities["system"]["available"])
        self.assertTrue(monitor.capabilities["gpu"]["available"])

    def test_auto_keeps_ipmi_when_rapl_unavailable(self):
        with patch.object(full, "_probe_ipmi_power", return_value=(6000, "probe succeeded")):
            monitor = create_monitor("auto", device_indices=[0])
        self.addCleanup(monitor.close)
        self.assertTrue(monitor.capabilities["system"]["available"])
        self.assertFalse(monitor.capabilities["cpu"]["available"])

    def test_strict_full_node_accepts_ipmi_without_intel_rapl(self):
        # Grace ARM can expose valid whole-node IPMI power without Intel RAPL,
        # even when the process runs as root. Only actual sensor reads matter.
        with patch.object(full, "_probe_ipmi_power", return_value=(441, "probe succeeded")):
            monitor = create_monitor("full_node", device_indices=[0])
        self.addCleanup(monitor.close)
        self.assertTrue(monitor.capabilities["system"]["available"])
        self.assertFalse(monitor.capabilities["cpu"]["available"])
        self.assertFalse(monitor.capabilities["dram"]["available"])
        self.assertIn("hardware/kernel support", monitor.capabilities["cpu"]["reason"])
        monitor._samples["gpu"] = [{"timestamp_s": t, "power_w": [100]} for t in (0, 1, 2)]
        monitor._samples["ipmi"] = [{"timestamp_s": t, "power_w": 441} for t in (0, 1, 2)]
        metrics = monitor.compute_metrics(2, 100, 1, start_time=0, end_time=2)
        self.assertEqual(metrics.system_energy_j, 882)
        self.assertEqual(metrics.total_energy_j, 882)
        self.assertEqual(metrics.energy_scope, "whole_node_ipmi")
        self.assertIsNone(metrics.cpu_energy_j)
        self.assertIsNone(metrics.dram_energy_j)

    def test_strict_full_node_rejects_missing_ipmi_even_with_cpu_rapl(self):
        self.domain("intel-rapl:0", "package-0")
        with patch.object(full, "_probe_ipmi_power", return_value=(None, "permission denied")), \
                self.assertRaisesRegex(RuntimeError, "IPMI: permission denied.*RAPL counters are optional"):
            full.FullNodeEnergyMonitor(device_indices=[0])
        self.nvml.nvmlShutdown.assert_called_once()

    def test_strict_full_node_rejects_missing_ipmi_without_cpu_rapl(self):
        with patch.object(full, "_probe_ipmi_power", return_value=(None, "IPMI unavailable")), \
                self.assertRaisesRegex(RuntimeError, "full_node requires a readable IPMI"):
            full.FullNodeEnergyMonitor(device_indices=[0])
        self.nvml.nvmlShutdown.assert_called_once()

    def test_high_ipmi_power_is_not_capped_and_is_whole_node_total(self):
        self.domain("intel-rapl:0", "package-0")
        self.domain("intel-rapl:0/intel-rapl:0:0", "dram")
        with patch.object(full, "_probe_ipmi_power", return_value=(6000, "probe succeeded")):
            monitor = self.full_monitor(strict=True)
        monitor._samples["gpu"] = [{"timestamp_s": t, "power_w": [100]} for t in (0, 1, 2)]
        monitor._samples["ipmi"] = [{"timestamp_s": t, "power_w": 6000} for t in (0, 1, 2)]
        monitor._samples["rapl"] = [
            {"timestamp_s": t, "energy_uj": {domain: t * (100_000_000 if kind == "cpu" else 20_000_000)
                                             for domain, kind in monitor._rapl.kinds.items()}}
            for t in (0, 1, 2)]
        metrics = monitor.compute_metrics(2, 100, 1, start_time=0, end_time=2)
        self.assertEqual(metrics.system_energy_j, 12000)
        self.assertEqual(metrics.total_energy_j, 12000)
        self.assertEqual(metrics.cpu_energy_j, 200)
        self.assertEqual(metrics.dram_energy_j, 40)
        self.assertEqual(metrics.components_energy_j, 440)
        self.assertEqual(metrics.energy_scope, "whole_node_ipmi")

    def test_ipmi_gaps_do_not_turn_into_complete_node_energy(self):
        with patch.object(full, "_probe_ipmi_power", return_value=(6000, "probe succeeded")):
            monitor = self.full_monitor()
        monitor._samples["gpu"] = [{"timestamp_s": t, "power_w": [100]} for t in (0, 1, 2)]
        monitor._samples["ipmi"] = [{"timestamp_s": 0, "power_w": 6000},
                                    {"timestamp_s": 1, "power_w": None},
                                    {"timestamp_s": 2, "power_w": 6000}]
        metrics = monitor.compute_metrics(2, 100, 1, start_time=0, end_time=2)
        self.assertIsNone(metrics.total_energy_j)
        self.assertEqual(metrics.gpu_energy_j, 200)
        self.assertEqual(metrics.energy_scope, "selected_gpus")

    def test_ipmi_probe_diagnostics_and_timeout(self):
        with patch.object(full.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, "Instantaneous power reading: 6500 Watts\n", "")) as run:
            self.assertEqual(full._probe_ipmi_power()[0], 6500)
            self.assertEqual(run.call_args.kwargs["timeout"], 5)
        for outcome, reason in [(FileNotFoundError(), "not installed"),
                                (PermissionError("denied"), "Cannot execute"),
                                (subprocess.TimeoutExpired("ipmitool", 5), "timed out")]:
            with patch.object(full.subprocess, "run", side_effect=outcome):
                self.assertIn(reason, full._probe_ipmi_power()[1])
        with patch.object(full.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            self.assertIsNone(full._probe_ipmi_power()[0])

    def test_ipmi_sample_keeps_complete_dcmi_output_from_one_query(self):
        stdout = (
            "Instantaneous power reading: 441 Watts\n"
            "Minimum during sampling period: 435 Watts\n"
            "Maximum during sampling period: 528 Watts\n"
            "Average power reading over sample period: 471 Watts\n"
            "IPMI timestamp: 09/14/26 02:03:39 UTC    Sampling period: 00000950 Seconds.\n"
            "Power reading state is: activated\n"
        )
        with patch.object(full, "_probe_ipmi_power", return_value=(441, "probe succeeded")):
            monitor = self.full_monitor()
        with patch.object(full.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout, "sensor diagnostic\n")) as command, \
                patch.object(full.time, "perf_counter", side_effect=[10.0, 10.08]):
            monitor._sample_ipmi()
        command.assert_called_once()
        sample = monitor.samples["ipmi"][0]
        self.assertEqual(sample["power_w"], 441)
        self.assertEqual(sample["timestamp_s"], 10.04)
        self.assertEqual(sample["acquisition_start_s"], 10.0)
        self.assertEqual(sample["acquisition_end_s"], 10.08)
        self.assertEqual(sample["dcmi"]["stdout"], stdout)
        self.assertEqual(sample["dcmi"]["stderr"], "sensor diagnostic\n")
        self.assertEqual(sample["dcmi"]["returncode"], 0)
        self.assertEqual(sample["dcmi"]["bmc_timestamp_raw"], "09/14/26 02:03:39 UTC")
        self.assertEqual(sample["dcmi"]["statistics_sampling_period_raw"], "00000950 Seconds.")
        self.assertEqual(sample["dcmi"]["average_power_raw"], "471 Watts")
        self.assertEqual(sample["dcmi"]["minimum_power_raw"], "435 Watts")
        self.assertEqual(sample["dcmi"]["maximum_power_raw"], "528 Watts")
        self.assertEqual(sample["dcmi"]["power_reading_state"], "activated")
        # Reported statistics period cannot establish sensor refresh/averaging.
        self.assertIsNone(monitor.capabilities["system"]["averaging_window_s"])
        json.dumps(sample, allow_nan=False)

    def test_ipmi_metadata_accepts_timestamp_and_period_on_separate_lines(self):
        metadata = full._dcmi_metadata(
            "IPMI timestamp: 09/14/26 02:03:39 UTC\n"
            "Sampling period: 00000950 Seconds.\nPower reading state is: activated\n")
        self.assertEqual(metadata["bmc_timestamp_raw"], "09/14/26 02:03:39 UTC")
        self.assertEqual(metadata["statistics_sampling_period_raw"], "00000950 Seconds.")

    def test_deactivated_ipmi_reading_is_unavailable_and_keeps_raw_evidence(self):
        stdout = ("Instantaneous power reading: 500 Watts\n"
                  "Average power reading over sample period: 450 Watts\n"
                  "Power reading state is: deactivated\n")
        with patch.object(full.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout, "")):
            reading = full._query_ipmi_power()
        self.assertIsNone(reading["power_w"])
        self.assertIn("deactivated", reading["reason"])
        self.assertEqual(reading["dcmi"]["stdout"], stdout)

    def test_failed_ipmi_queries_keep_partial_output_without_using_its_power(self):
        stdout = "Instantaneous power reading: 500 Watts\n"
        with patch.object(full.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, stdout, "failed")):
            reading = full._query_ipmi_power()
        self.assertIsNone(reading["power_w"])
        self.assertEqual(reading["dcmi"]["stdout"], stdout)
        self.assertEqual(reading["dcmi"]["stderr"], "failed")
        with patch.object(full.subprocess, "run", side_effect=subprocess.TimeoutExpired(
                "ipmitool", 5, output=stdout.encode(), stderr=b"partial failure")):
            reading = full._query_ipmi_power()
        self.assertIsNone(reading["power_w"])
        self.assertEqual(reading["dcmi"]["stdout"], stdout)
        self.assertEqual(reading["dcmi"]["stderr"], "partial failure")
        json.dumps(reading, allow_nan=False)

    def test_flat_ipmi_window_warns_but_retains_measured_energy(self):
        with patch.object(full, "_probe_ipmi_power", return_value=(515, "probe succeeded")):
            monitor = self.full_monitor()
        monitor._samples["gpu"] = [{"timestamp_s": t, "power_w": [100]} for t in (0, 1, 2, 3, 4)]
        # Changes outside the integration window must not hide flat readings.
        monitor._samples["ipmi"] = [{"timestamp_s": t, "power_w": w}
                                    for t, w in [(0, 400), (1, 515), (2, 515), (3, 515), (4, 600)]]
        metrics = monitor.compute_metrics(2, 100, 1, start_time=1, end_time=3)
        self.assertEqual(metrics.system_energy_j, 1030)
        self.assertTrue(any("all readings covering this window are identical (515 W)" in w
                            for w in metrics.warnings))
        # A varying interpolation endpoint means the measured window is not flat.
        metrics = monitor.compute_metrics(2.5, 100, 1, start_time=0.5, end_time=3)
        self.assertFalse(any("identical" in w for w in metrics.warnings))


if __name__ == "__main__":
    unittest.main()

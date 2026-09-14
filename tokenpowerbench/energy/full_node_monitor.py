"""Optional CPU package/DRAM RAPL and whole-node IPMI measurements.

Access is probed by reading sensors, without sudo, chmod, or root assumptions.
Whole-node measurement requires IPMI; CPU and DRAM RAPL remain independent,
optional components, including on Grace ARM systems without Intel RAPL.
RAPL package counters and DRAM counters are retained; core/uncore/psys domains
are excluded to avoid adding overlapping counters to CPU package energy.
"""

import bisect
import math
import os
import re
import subprocess
import time

from .gpu_monitor import GPUEnergyMonitor, integrate_power

_RAPL_ROOT = "/sys/class/powercap/intel-rapl"
_CPU_SAMPLE_INTERVAL = 0.5
_IPMI_SAMPLE_INTERVAL = 1.0
_IPMI_TIMEOUT = 5


def integrate_energy_counter(readings, start_time, end_time, max_energy_uj=None):
    """Interpolate cumulative RAPL counters at the exact window boundaries."""
    result = {"energy_j": None, "average_power_w": None, "valid": False,
              "sample_count": sum(start_time <= t <= end_time for t, _ in readings), "reason": None}
    if not (math.isfinite(start_time) and math.isfinite(end_time)) or end_time <= start_time:
        raise ValueError("Energy window must have finite timestamps and positive duration")
    if len(readings) < 2 or readings[0][0] > start_time or readings[-1][0] < end_time:
        result["reason"] = "RAPL samples do not cover the complete measurement window"
        return result
    if any(not math.isfinite(t) for t, _ in readings) or any(
            b[0] <= a[0] for a, b in zip(readings, readings[1:])):
        result["reason"] = "RAPL sample timestamps are invalid"
        return result
    if result["sample_count"] < 2 or end_time - start_time < _CPU_SAMPLE_INTERVAL:
        result["reason"] = "insufficient RAPL sampling resolution for this window"
        return result
    energy_uj = 0.0
    for (t0, e0), (t1, e1) in zip(readings, readings[1:]):
        lo, hi = max(t0, start_time), min(t1, end_time)
        if hi <= lo:
            continue
        if any(e is None or not math.isfinite(e) or e < 0 for e in (e0, e1)):
            result["reason"] = "missing or invalid RAPL energy counter in measurement window"
            return result
        delta = e1 - e0
        if delta < 0:
            if max_energy_uj is None or max_energy_uj <= 0:
                result["reason"] = "RAPL counter wrapped but max_energy_range_uj is unavailable"
                return result
            delta += max_energy_uj
            if delta < 0:
                result["reason"] = "invalid RAPL counter wraparound"
                return result
        energy_uj += delta * (hi - lo) / (t1 - t0)
    energy_j = energy_uj / 1_000_000
    result.update(energy_j=energy_j, average_power_w=energy_j / (end_time - start_time), valid=True)
    return result


class _RaplReader:
    def __init__(self):
        self.domains = {}
        self.kinds = {}
        self._max_energy = {}
        self.errors = []
        self.last_errors = {}
        self.unreadable_kinds = set()
        root = _RAPL_ROOT
        if root == "/sys/class/powercap/intel-rapl" and not os.path.isdir(root):
            root = "/sys/class/powercap"
        if not os.path.isdir(root):
            self.errors.append("Intel RAPL sysfs interface not found; availability depends on "
                               "hardware/kernel support, and root access cannot supply a missing interface")
            return
        pending, visited = [root], set()
        while pending:
            directory = pending.pop()
            realpath = os.path.realpath(directory)
            if realpath in visited:
                continue
            visited.add(realpath)
            try:
                with os.scandir(directory) as entries:
                    pending.extend(entry.path for entry in entries if entry.name.startswith("intel-rapl:"))
            except OSError as exc:
                self.errors.append(f"Cannot list RAPL domains: {exc}")
            if directory == root:
                continue
            try:
                with open(os.path.join(directory, "name")) as stream:
                    name = stream.read().strip()
            except (OSError, ValueError) as exc:
                self.errors.append(f"Cannot read RAPL domain name: {exc}")
                continue
            kind = "cpu" if re.fullmatch(r"package-\d+", name) else "dram" if name.lower() == "dram" else None
            if kind is None:
                continue
            key = f"{os.path.basename(directory)}:{name}"
            try:
                with open(os.path.join(directory, "energy_uj")) as stream:
                    energy = int(stream.read().strip())
                if energy < 0:
                    raise ValueError("negative energy counter")
            except (OSError, ValueError) as exc:
                self.errors.append(f"{key} unavailable: {exc}")
                self.unreadable_kinds.add(kind)
                continue
            self.domains[key] = directory
            self.kinds[key] = kind
            try:
                with open(os.path.join(directory, "max_energy_range_uj")) as stream:
                    maximum = int(stream.read().strip())
                self._max_energy[key] = maximum if maximum > 0 else None
            except (OSError, ValueError):
                self._max_energy[key] = None

    def capability(self, kind):
        domains = [name for name in self.domains if self.kinds[name] == kind]
        available = bool(domains) and kind not in self.unreadable_kinds
        reason = ("RAPL energy counter read succeeded" if available else
                  "; ".join(self.errors) or f"No compatible Intel RAPL {kind} domains found; "
                  "these optional counters depend on hardware/kernel support")
        return {"available": available, "reason": reason, "domains": domains,
                "scope": "host_cpu_packages" if kind == "cpu" else "host_dram_domains",
                "sampling_interval_s": _CPU_SAMPLE_INTERVAL,
                "minimum_window_s": _CPU_SAMPLE_INTERVAL,
                "source": "RAPL energy_uj counter differences",
                "discovery_warnings": list(self.errors)}

    def read_energy(self):
        readings = {}
        for name, directory in self.domains.items():
            try:
                with open(os.path.join(directory, "energy_uj")) as stream:
                    value = int(stream.read().strip())
                readings[name] = value if value >= 0 else None
            except (OSError, ValueError) as exc:
                readings[name] = None
                self.last_errors[name] = str(exc)
        return readings


def _dcmi_metadata(stdout):
    """Retain BMC-reported context without treating statistics as refresh rate."""
    def field(label):
        match = re.search(r"^[ \t]*" + re.escape(label) + r"[ \t]*:[ \t]*([^\r\n]*)",
                          stdout, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else None

    timestamp = field("IPMI timestamp")
    sampling_period = field("Sampling period")
    if timestamp is not None:
        # ipmitool may print the timestamp and sampling period on one line.
        parts = re.split(r"[ \t]+Sampling period[ \t]*:", timestamp, maxsplit=1, flags=re.IGNORECASE)
        timestamp = parts[0].strip()
        if len(parts) == 2:
            sampling_period = parts[1].strip()
    return {
        "bmc_timestamp_raw": timestamp,
        "statistics_sampling_period_raw": sampling_period,
        "power_reading_state": field("Power reading state is"),
        "minimum_power_raw": field("Minimum during sampling period"),
        "maximum_power_raw": field("Maximum during sampling period"),
        "average_power_raw": field("Average power reading over sample period"),
    }


def _query_ipmi_power():
    """Issue one query and return its power, diagnostics, and complete output."""
    reading = {"power_w": None, "reason": None,
               "dcmi": {"stdout": "", "stderr": "", "returncode": None}}
    details = reading["dcmi"]
    try:
        result = subprocess.run(["ipmitool", "dcmi", "power", "reading"],
                                capture_output=True, text=True, timeout=_IPMI_TIMEOUT)
    except FileNotFoundError:
        reading["reason"] = "ipmitool is not installed"
        return reading
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired may contain bytes even with text=True.
        for name, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            details[name] = value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        details.update(_dcmi_metadata(details["stdout"]))
        reading["reason"] = f"IPMI power query timed out after {_IPMI_TIMEOUT} seconds"
        return reading
    except OSError as exc:
        reading["reason"] = f"Cannot execute ipmitool: {exc}"
        return reading
    details.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    details.update(_dcmi_metadata(result.stdout))
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        reading["reason"] = f"IPMI query failed (exit {result.returncode}): {detail[:400]}"
        return reading
    if (details["power_reading_state"] or "").lower() == "deactivated":
        reading["reason"] = "IPMI reports that the power reading state is deactivated"
        return reading
    for line in result.stdout.splitlines():
        if "Instantaneous power reading" in line:
            try:
                value = float(line.split(":", 1)[1].strip().split()[0])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("power must be finite and nonnegative")
                reading.update(power_w=value, reason="IPMI DCMI power reading succeeded")
            except (ValueError, IndexError):
                reading["reason"] = "IPMI returned an invalid instantaneous power reading"
            return reading
    reading["reason"] = "IPMI response has no instantaneous power reading"
    return reading


def _probe_ipmi_power():
    """Return the existing (watts, reason) probe contract from one query."""
    reading = _query_ipmi_power()
    return reading["power_w"], reading["reason"]


def _ipmi_available():
    return _probe_ipmi_power()[0] is not None


def _read_ipmi_power():
    return _probe_ipmi_power()[0]


class FullNodeEnergyMonitor(GPUEnergyMonitor):
    def __init__(self, device_indices=None, device_uuids=None, strict=True):
        super().__init__(device_indices=device_indices, device_uuids=device_uuids)
        try:
            self._rapl = _RaplReader()
            self._capabilities["cpu"] = self._rapl.capability("cpu")
            self._capabilities["dram"] = self._rapl.capability("dram")
            ipmi_power, ipmi_reason = _probe_ipmi_power()
            self._capabilities["system"] = {
                "available": ipmi_power is not None, "reason": ipmi_reason,
                "scope": "whole_node_ipmi", "sampling_interval_s": _IPMI_SAMPLE_INTERVAL,
                "minimum_window_s": _IPMI_SAMPLE_INTERVAL, "averaging_window_s": None,
                "source": "ipmitool dcmi power reading",
                "timestamp_clock": "time.perf_counter",
                "timestamp_definition": "midpoint of command acquisition; BMC sensor age is unknown",
                "warning": "BMC power update/averaging resolution is hardware-dependent; 1-second polling is a lower bound, not proof of phase resolution.",
            }
            if strict and ipmi_power is None:
                raise RuntimeError("full_node requires a readable IPMI whole-node power sensor. "
                                   f"IPMI: {ipmi_reason}. CPU/DRAM Intel RAPL counters are optional. "
                                   "Use --monitor auto or --monitor gpu_only when these sensors are unavailable.")
        except Exception:
            self.close()
            raise

    def _samplers(self):
        samplers = super()._samplers()
        if self._capabilities["cpu"]["available"] or self._capabilities["dram"]["available"]:
            samplers.append((self._sample_rapl, _CPU_SAMPLE_INTERVAL))
        if self._capabilities["system"]["available"]:
            samplers.append((self._sample_ipmi, _IPMI_SAMPLE_INTERVAL))
        return samplers

    def _sample_rapl(self):
        values = self._rapl.read_energy()
        for domain, error in self._rapl.last_errors.items():
            self._sample_errors[f"{self._rapl.kinds[domain]}:{domain}"] = error
        sample = {"timestamp_s": time.perf_counter(), "energy_uj": values}
        with self._lock:
            self._samples["rapl"].append(sample)

    def _sample_ipmi(self):
        acquisition_start = time.perf_counter()
        reading = _query_ipmi_power()
        acquisition_end = time.perf_counter()
        sample = {"timestamp_s": (acquisition_start + acquisition_end) / 2,
                  "acquisition_start_s": acquisition_start,
                  "acquisition_end_s": acquisition_end, **reading}
        with self._lock:
            if reading["power_w"] is None:
                self._sample_errors["system"] = reading["reason"]
            self._samples["ipmi"].append(sample)

    def compute_metrics(self, duration, total_output_tokens, num_responses,
                        start_time=None, end_time=None):
        metrics = super().compute_metrics(duration, total_output_tokens, num_responses,
                                          start_time=start_time, end_time=end_time)
        start_time, end_time = metrics.start_time, metrics.end_time
        samples = self.samples
        for kind in ("cpu", "dram"):
            capability = self._capabilities[kind]
            if not capability["available"]:
                metrics.warnings.append(f"{kind.upper()}: {capability['reason']}")
                continue
            energies = []
            for domain in capability["domains"]:
                integrated = integrate_energy_counter(
                    [(s["timestamp_s"], s["energy_uj"].get(domain)) for s in samples["rapl"]],
                    start_time, end_time, self._rapl._max_energy.get(domain))
                metrics.sample_counts[domain] = integrated["sample_count"]
                if not integrated["valid"]:
                    metrics.warnings.append(f"RAPL {domain}: {integrated['reason']}")
                energies.append(integrated["energy_j"])
            if energies and all(value is not None for value in energies):
                energy = sum(energies)
                setattr(metrics, f"{kind}_energy_j", energy)
                setattr(metrics, f"{kind}_avg_power_w", energy / metrics.duration)
        if self._capabilities["system"]["available"]:
            integrated = integrate_power([(s["timestamp_s"], s["power_w"]) for s in samples["ipmi"]],
                                         start_time, end_time)
            if metrics.duration < _IPMI_SAMPLE_INTERVAL:
                integrated.update(energy_j=None, average_power_w=None, valid=False,
                                  reason="window shorter than the 1-second IPMI sampling interval")
            metrics.sample_counts["system"] = integrated["sample_count"]
            metrics.system_energy_j = integrated["energy_j"]
            metrics.system_avg_power_w = integrated["average_power_w"]
            if not integrated["valid"]:
                metrics.warnings.append(f"IPMI: {integrated['reason']}")
            else:
                timestamps = [s["timestamp_s"] for s in samples["ipmi"]]
                left = max(0, bisect.bisect_right(timestamps, start_time) - 1)
                right = bisect.bisect_left(timestamps, end_time) + 1
                powers = [s["power_w"] for s in samples["ipmi"][left:right]]
                if len(powers) >= 2 and len(set(powers)) == 1:
                    metrics.warnings.append(
                        f"IPMI: all readings covering this window are identical ({powers[0]:g} W); "
                        "stable power or slow/stale BMC updates can cause this. Inspect raw DCMI "
                        "timestamps and metadata before interpreting phase energy."
                    )
            metrics.warnings.append(self._capabilities["system"]["warning"])
        else:
            metrics.warnings.append(f"IPMI: {self._capabilities['system']['reason']}")
        if metrics.system_energy_j is not None:
            metrics.energy_scope = "whole_node_ipmi"
        elif metrics.cpu_energy_j is not None or metrics.dram_energy_j is not None:
            metrics.energy_scope = "available_components"
        return metrics

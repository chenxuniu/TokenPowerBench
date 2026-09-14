"""Exercise launcher control flow with fake scheduler, Ray, and Python programs."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOCK = r'''
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

command = Path(sys.argv[0]).name
arguments = sys.argv[1:]
log = Path(os.environ["TPB_MOCK_LOG"])
record = {"command": command, "arguments": arguments,
          "node": os.environ.get("SLURMD_NODENAME"),
          "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "cache": os.environ.get("VLLM_CACHE_ROOT"),
          "head_port": os.environ.get("RAY_HEAD_PORT"),
          "head_address": os.environ.get("RAY_HEAD_ADDRESS"),
          "ray_address": os.environ.get("RAY_ADDRESS"),
          "interface": os.environ.get("TPB_NETWORK_INTERFACE")}
def append(value):
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.write(descriptor, (json.dumps(value) + "\n").encode())
    os.close(descriptor)
def entries():
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
append(record)
if command == "sbatch":
    print("12345;cluster")
elif command == "scontrol":
    print("head-node\nworker-node")
elif command == "srun":
    node = next(arg.split("=",1)[1] for arg in arguments if arg.startswith("--nodelist="))
    os.environ["SLURMD_NODENAME"] = node
    start = arguments.index("bash")
    os.execvp("bash", arguments[start:])
elif command == "timeout":
    os.execv(arguments[1], arguments[1:])
elif command == "ray":
    if arguments[0] != "start":
        raise SystemExit("Global Ray commands are forbidden in this mock")
    def terminate(signum, frame):
        append({"command": "ray-terminated", "node": os.environ["SLURMD_NODENAME"]})
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    while True:
        time.sleep(.02)
elif command == "python":
    if arguments[0] == "-c":
        print(os.environ.get("TPB_MOCK_GPUS", "4"))
    elif arguments[0] == "-":
        source = sys.stdin.read()
        if "load_profile" in source:
            sys.argv = arguments
            exec(compile(source, "<launcher-profile>", "exec"))
        elif "server.bind" in source:
            if os.environ.get("TPB_MOCK_PORT_BUSY"):
                raise SystemExit(3)
        elif "hostname, interface" in source:
            if os.environ.get("TPB_MOCK_BAD_IP"):
                raise SystemExit(8)
            print("10.0.0.1" if arguments[1] == "head-node" else "10.0.0.2")
        else:
            count = int(arguments[2]) if "expected_nodes" in source else 1
            deadline = time.monotonic() + 5
            while sum(e["command"] == "ray" for e in entries()) < count:
                if time.monotonic() > deadline:
                    raise SystemExit("Fake Ray services did not start")
                time.sleep(.02)
            if "expected_nodes" in source:
                marker = arguments[4]
                ray = types.ModuleType("ray")
                ray.init = lambda **kwargs: None
                ray.shutdown = lambda: None
                def nodes():
                    if count == 2 and os.environ.get("TPB_MOCK_READY_FAIL"):
                        raise RuntimeError("Simulated Ray readiness failure")
                    result = []
                    for service in (e for e in entries() if e["command"] == "ray"):
                        resources = json.loads(next((arg.split("=", 1)[1] for arg in service["arguments"] if arg.startswith("--resources=")), "{}"))
                        resources["GPU"] = 4
                        if os.environ.get("TPB_MOCK_UNRELATED_HEAD"):
                            resources.pop(marker, None)
                        result.append({"Alive": True, "Resources": resources})
                    return result
                ray.nodes = nodes
                sys.modules["ray"] = ray
                sys.argv = arguments
                exec(compile(source, "<launcher-readiness>", "exec"))
    else:
        append({"command": "benchmark", "arguments": arguments})
        if os.environ.get("TPB_MOCK_HOLD_BENCHMARK"):
            def stop_benchmark(signum, frame):
                append({"command": "benchmark-terminated"})
                raise SystemExit(0)
            signal.signal(signal.SIGTERM, stop_benchmark)
            while True:
                time.sleep(.02)
        raise SystemExit(int(os.environ.get("TPB_MOCK_BENCHMARK_EXIT", "0")))
else:
    raise SystemExit("Unexpected mock command " + command)
'''


class SlurmLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tpb shell test ")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.project = self.path / "shared project"
        shutil.copytree(ROOT / "scripts", self.project / "scripts")
        shutil.copytree(ROOT / "MultipleNode", self.project / "MultipleNode")
        shutil.copytree(ROOT / "tokenpowerbench", self.project / "tokenpowerbench")
        (self.project / "run_multi_node.py").write_text("# The fake interpreter records invocations.\n")
        self.bin = self.path / "environment" / "bin"
        self.bin.mkdir(parents=True)
        self.log = self.path / "commands.jsonl"
        for name in ("sbatch", "scontrol", "srun", "timeout", "ray", "python"):
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n" + MOCK)
            path.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("TPB_", "SLURM_", "SLURMD_", "RAY_", "VLLM_"))}
        self.env.update(PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        TPB_PYTHON=str(self.bin / "python"), TPB_PROJECT_DIR=str(self.project),
                        TPB_LOG_DIR=str(self.path / "logs"), TPB_MOCK_LOG=str(self.log),
                        TPB_GPUS_PER_NODE="4", TPB_CPUS_PER_NODE="16",
                        SLURM_TMPDIR=str(self.path / "node cache"),
                        CUDA_VISIBLE_DEVICES="GPU-aaa,GPU-bbb,GPU-ccc,GPU-ddd")

    def records(self, command):
        if not self.log.exists():
            return []
        return [record for line in self.log.read_text().splitlines()
                if (record := json.loads(line))["command"] == command]

    def invoke(self, script, *arguments, allocation=False):
        if allocation:
            self.env.update(SLURM_JOB_ID="12345", SLURM_JOB_NODELIST="head-node,worker-node",
                            SLURMD_NODENAME="head-node", SLURM_CPUS_PER_TASK="16")
        return subprocess.run(["bash", str(self.project / script), *arguments], env=self.env,
                              cwd=self.path, text=True, capture_output=True, timeout=20)

    def profile(self, **overrides):
        path = self.path / "cluster profiles" / "slurm cluster.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"schema_version": 1, "mode": "slurm", "head_address": None,
                                    "head_port": 6388, "worker_addresses": [],
                                    "network_interface": "ib0", **overrides}))
        return path

    def test_submission_is_one_allocation_and_preserves_arguments(self):
        self.env.update(TPB_PARTITION="accelerated", TPB_ACCOUNT="research")
        result = self.invoke("scripts/submit_jobs.sh", "3", "--", "--models", "model with spaces", "--output-dir", "result path")
        self.assertEqual(result.returncode, 0, result.stderr)
        submitted = self.records("sbatch")
        self.assertEqual(len(submitted), 1)
        args = submitted[0]["arguments"]
        self.assertIn("--nodes=4", args)
        self.assertIn("--exclusive", args)
        self.assertIn("--partition=accelerated", args)
        self.assertEqual(args[-4:], ["--models", "model with spaces", "--output-dir", "result path"])
        self.assertTrue((self.path / "logs").is_dir())

    def test_bad_worker_count_does_not_submit(self):
        result = self.invoke("scripts/submit_jobs.sh", "-1", "--models", "small")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.records("sbatch"))

    def test_interactive_wizard_is_rejected_before_submission_or_service_startup(self):
        for script, leading in (("scripts/submit_jobs.sh", ("1",)), ("scripts/slurm_head.sh", ())):
            with self.subTest(script=script):
                result = self.invoke(script, *leading, "--configure-cluster", allocation=not leading)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("interactively before submitting", result.stderr)
                self.assertFalse(self.records("sbatch"))
                self.assertFalse(self.records("srun"))
                self.assertFalse(self.records("ray"))

    def test_submission_applies_profile_and_preserves_relative_path_with_spaces(self):
        profile = self.profile()
        self.env.update(RAY_HEAD_PORT="6400", TPB_NETWORK_INTERFACE="eth0",
                        RAY_HEAD_ADDRESS="10.9.9.9", RAY_ADDRESS="10.9.9.9:6400")
        result = self.invoke("scripts/submit_jobs.sh", "1", "--models", "model with spaces",
                             "--cluster-config", str(profile.relative_to(self.path)))
        self.assertEqual(result.returncode, 0, result.stderr)
        submitted = self.records("sbatch")[0]
        self.assertEqual(submitted["head_port"], "6388")
        self.assertEqual(submitted["interface"], "ib0")
        self.assertIsNone(submitted["head_address"])
        self.assertIsNone(submitted["ray_address"])
        self.assertEqual(submitted["arguments"][-4:], ["--models", "model with spaces",
                                                       "--cluster-config", str(profile.resolve())])

    def test_controller_profile_port_interface_and_cli_override_reach_all_nodes(self):
        profile = self.profile()
        self.env.update(RAY_HEAD_ADDRESS="10.9.9.9", RAY_ADDRESS="10.9.9.9:6400")
        result = self.invoke("scripts/slurm_head.sh", "--models", "small",
                             "--cluster-config=" + str(profile), "--ray-head-port=6399", allocation=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        services = self.records("ray")
        self.assertEqual(len(services), 2)
        for service in services:
            self.assertEqual(service["head_port"], "6399")
            self.assertEqual(service["interface"], "ib0")
            self.assertEqual(service["head_address"], "10.0.0.1")
            self.assertEqual(service["ray_address"], "10.0.0.1:6399")
            if service["node"] == "head-node":
                self.assertIn("--port=6399", service["arguments"])
                self.assertIn("--node-ip-address=10.0.0.1", service["arguments"])
            else:
                self.assertIn("--address=10.0.0.1:6399", service["arguments"])
        ip_probes = [record for record in self.records("python")
                     if record["arguments"][1:2] in (["head-node"], ["worker-node"])]
        self.assertTrue(ip_probes)
        self.assertTrue(all(probe["arguments"][-1] == "ib0" for probe in ip_probes))
        self.assertEqual(self.records("benchmark")[0]["arguments"][-4:],
                         ["--ray-head-address", "10.0.0.1", "--ray-head-port", "6399"])

    def test_controller_uses_profile_port_and_automatic_interface_over_stale_environment(self):
        profile = self.profile(network_interface=None)
        self.env.update(RAY_HEAD_PORT="6400", TPB_NETWORK_INTERFACE="eth0")
        result = self.invoke("scripts/slurm_head.sh", "--models", "small",
                             "--cluster-config", str(profile), allocation=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for service in self.records("ray"):
            self.assertEqual(service["head_port"], "6388")
            self.assertEqual(service["interface"], "")

    def test_invalid_missing_and_manual_profiles_fail_before_submission_or_startup(self):
        missing = self.path / "missing cluster.json"
        malformed = self.path / "malformed.json"
        malformed.write_text("not json")
        profiles = [missing, malformed]
        for name, data in (
                ("manual", {"schema_version": 1, "mode": "manual", "head_address": "10.1.2.3",
                            "head_port": 6388, "worker_addresses": ["10.1.2.4"], "network_interface": None}),
                ("invalid", {"schema_version": 1, "mode": "slurm", "head_address": None,
                             "head_port": 70000, "worker_addresses": [], "network_interface": "ib0"})):
            path = self.path / f"{name}.json"
            path.write_text(json.dumps(data))
            profiles.append(path)
        for script in ("scripts/submit_jobs.sh", "scripts/slurm_head.sh"):
            for profile in profiles:
                with self.subTest(script=script, profile=profile.name):
                    before = len(self.records("python"))
                    leading = ("1",) if script.endswith("submit_jobs.sh") else ()
                    result = self.invoke(script, *leading, "--models", "small", "--cluster-config", str(profile),
                                         allocation=not leading)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.records("sbatch"))
                    self.assertFalse(self.records("srun"))
                    self.assertFalse(self.records("ray"))
                    self.assertFalse(self.records("benchmark"))
                    self.assertFalse(any(record["arguments"][0] == "-c"
                                         for record in self.records("python")[before:]))

    def test_cli_port_override_without_profile_is_validated_before_submission(self):
        self.env["RAY_HEAD_PORT"] = "6380"
        result = self.invoke("scripts/submit_jobs.sh", "1", "--models", "small", "--ray-head-port", "6390")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.records("sbatch")[0]["head_port"], "6390")
        for invalid in ("0", "65536", "invalid"):
            with self.subTest(port=invalid):
                result = self.invoke("scripts/submit_jobs.sh", "1", "--models", "small", "--ray-head-port", invalid)
                self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.records("sbatch")), 1)

    def test_explicit_head_address_is_rejected_before_submission_or_startup(self):
        profile = self.profile()
        for script, leading in (("scripts/submit_jobs.sh", ("1",)), ("scripts/slurm_head.sh", ())):
            for profile_args in ((), ("--cluster-config", str(profile))):
                with self.subTest(script=script, profile=bool(profile_args)):
                    result = self.invoke(script, *leading, "--models", "small", *profile_args,
                                         "--ray-head-address=10.0.0.1", allocation=not leading)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("use tpbench-multi directly", result.stderr)
                    self.assertFalse(self.records("sbatch"))
                    self.assertFalse(self.records("srun"))
                    self.assertFalse(self.records("ray"))
                    self.assertFalse(self.records("benchmark"))

    def test_no_profile_preserves_inherited_head_address_consistency_check(self):
        self.env.update(RAY_HEAD_ADDRESS="10.9.9.9", RAY_ADDRESS="10.9.9.9:6400")
        result = self.invoke("scripts/slurm_head.sh", "--models", "small", allocation=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RAY_HEAD_ADDRESS differs", result.stderr)
        self.assertFalse(self.records("srun"))
        self.assertFalse(self.records("ray"))

    def test_controller_starts_allocated_nodes_and_cleans_only_its_services(self):
        result = self.invoke("scripts/slurm_head.sh", "--models", "model with spaces", allocation=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.records("srun")), 2)
        services = self.records("ray")
        self.assertEqual({service["node"] for service in services}, {"head-node", "worker-node"})
        for service in services:
            self.assertIn("--block", service["arguments"])
            self.assertIn("--num-gpus=4", service["arguments"])
            self.assertEqual(service["cuda"], self.env["CUDA_VISIBLE_DEVICES"])
            self.assertIn("tpbench-12345-0-" + service["node"], service["cache"])
        self.assertEqual(len(self.records("ray-terminated")), 2)
        benchmark = self.records("benchmark")[0]["arguments"]
        self.assertEqual(benchmark[1:3], ["--models", "model with spaces"])
        self.assertEqual(benchmark[-4:], ["--ray-head-address", "10.0.0.1", "--ray-head-port", "6379"])

    def test_benchmark_failure_preserves_exit_code_and_closes_services(self):
        self.env["TPB_MOCK_BENCHMARK_EXIT"] = "37"
        result = self.invoke("scripts/slurm_head.sh", "--models", "small", allocation=True)
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual(len(self.records("ray-terminated")), 2)

    def test_readiness_failure_never_runs_benchmark(self):
        self.env["TPB_MOCK_READY_FAIL"] = "1"
        result = self.invoke("scripts/slurm_head.sh", "--models", "small", allocation=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(self.records("benchmark"))
        self.assertEqual(len(self.records("ray-terminated")), 2)

    def test_visibility_mismatch_fails_before_startup(self):
        self.env["TPB_MOCK_GPUS"] = "2"
        result = self.invoke("scripts/slurm_head.sh", "--models", "small", allocation=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs", result.stderr)
        self.assertFalse(self.records("srun"))

    def test_address_failure_is_not_hidden_by_export(self):
        self.env["TPB_MOCK_BAD_IP"] = "1"
        result = self.invoke("scripts/slurm_head.sh", "--models", "small", allocation=True)
        self.assertEqual(result.returncode, 8)
        self.assertFalse(self.records("srun"))

    def test_worker_requires_controller_address(self):
        result = self.invoke("scripts/slurm_worker.sh", "worker", allocation=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RAY_HEAD_ADDRESS", result.stderr)
        self.assertFalse(self.records("ray"))

    def test_occupied_head_port_does_not_launch_or_join_ray(self):
        self.env["TPB_MOCK_PORT_BUSY"] = "1"
        result = self.invoke("scripts/slurm_head.sh", "--models", "small", allocation=True)
        self.assertEqual(result.returncode, 3)
        self.assertFalse(self.records("srun"))
        self.assertFalse(self.records("benchmark"))

    def test_unrelated_head_marker_is_rejected_before_workers_join(self):
        self.env["TPB_MOCK_UNRELATED_HEAD"] = "1"
        result = self.invoke("scripts/slurm_head.sh", "--models", "small", allocation=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("does not belong to this launcher", result.stderr)
        self.assertEqual(len(self.records("srun")), 1)
        self.assertFalse(self.records("benchmark"))
        self.assertEqual(len(self.records("ray-terminated")), 1)

    def test_controller_signal_terminates_driver_and_owned_services(self):
        self.env.update(TPB_MOCK_HOLD_BENCHMARK="1", SLURM_JOB_ID="12345",
                        SLURM_JOB_NODELIST="head-node,worker-node", SLURMD_NODENAME="head-node",
                        SLURM_CPUS_PER_TASK="16")
        process = subprocess.Popen(["bash", str(self.project / "scripts/slurm_head.sh"), "--models", "small"],
                                   env=self.env, cwd=self.path, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not self.records("benchmark"):
                if process.poll() is not None or time.monotonic() > deadline:
                    self.fail("Benchmark did not start before signal test")
                time.sleep(.05)
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 143, stderr)
            self.assertEqual(len(self.records("benchmark-terminated")), 1)
            self.assertEqual(len(self.records("ray-terminated")), 2)
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=20)

    def test_source_entrypoints_delegate_to_canonical_submission(self):
        for script in ("MultipleNode/submit_ray_jobs.sh", "MultipleNode/run_ray_head.sh"):
            result = self.invoke(script, "1", "--models", "small")
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.records("sbatch")), 2)


if __name__ == "__main__":
    unittest.main()

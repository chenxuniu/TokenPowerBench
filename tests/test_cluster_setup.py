"""Terminal cluster configuration without Ray, remote nodes, or Slurm services."""

import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tokenpowerbench.distributed import cluster_setup as setup


def profile(mode="manual", **values):
    return {"schema_version": 1, "mode": mode,
            "head_address": "192.0.2.10" if mode == "manual" else None,
            "head_port": 6379, "worker_addresses": [], "network_interface": None, **values}


def interface_data(*addresses):
    return json.dumps([{"ifname": "ib0", "addr_info": [
        {"family": "inet", "scope": "global", "local": address} for address in addresses]}])


class ClusterSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "cluster.json"
        self.allocation = {"SLURM_JOB_ID": "123", "SLURM_JOB_NODELIST": "gpu[01-03]"}

    def wizard(self, answers, env=None):
        output = io.StringIO()
        with patch("builtins.input", side_effect=answers), patch("sys.stdout", output), \
                patch.dict(os.environ, env or {}, clear=True):
            code = setup.configure(self.path)
        return code, output.getvalue()

    def test_valid_profiles_are_copied_with_exact_schema(self):
        original = profile(worker_addresses=["192.0.2.11"])
        validated = setup.validate_profile(original)
        self.assertEqual(validated, original)
        validated["worker_addresses"].append("192.0.2.12")
        self.assertEqual(original["worker_addresses"], ["192.0.2.11"])
        self.assertEqual(setup.validate_profile(profile("slurm", network_interface="ib0")),
                         profile("slurm", network_interface="ib0"))

    def test_profile_schema_and_numeric_types_are_strict(self):
        invalid = [None, [], {}, profile(schema_version=True), profile(schema_version=2),
                   profile(schema_version=1.0), profile(mode="auto"), profile(mode=[]),
                   profile(extra="ignored"), profile(head_port="6379"), profile(head_port=True),
                   profile(head_port=0), profile(head_port=65536), profile(head_port=1.5)]
        missing = profile()
        del missing["network_interface"]
        invalid.append(missing)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                setup.validate_profile(value)

    def test_manual_addresses_and_workers_are_validated(self):
        for changes in ({"head_address": "head.example"}, {"head_address": "::1"},
                        {"head_address": 123}, {"head_address": "192.0.2.999"},
                        {"worker_addresses": "192.0.2.11"}, {"worker_addresses": ("192.0.2.11",)},
                        {"worker_addresses": ["192.0.2.11", "192.0.2.11"]},
                        {"worker_addresses": ["192.0.2.10"]}, {"worker_addresses": [None]},
                        {"network_interface": "ib0"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                setup.validate_profile(profile(**changes))

    def test_manual_addresses_must_be_usable_cluster_peers(self):
        for address in ("0.0.0.0", "127.0.0.1", "169.254.0.1", "224.0.0.1", "240.0.0.1", "255.255.255.255"):
            for field in ("head_address", "worker_addresses"):
                with self.subTest(address=address, field=field), self.assertRaisesRegex(ValueError, "usable IPv4 peer"):
                    setup.validate_profile(profile(**{field: address if field == "head_address" else [address]}))
        for address in ("10.0.0.1", "172.16.0.1", "192.168.0.1", "192.0.2.10"):
            self.assertEqual(setup.validate_profile(profile(head_address=address))["head_address"], address)

    def test_slurm_profiles_never_store_allocation_nodes_and_require_safe_interface(self):
        for changes in ({"head_address": "192.0.2.10"}, {"worker_addresses": ["192.0.2.11"]},
                        {"network_interface": ""}, {"network_interface": "--help"},
                        {"network_interface": "ib0; hostname"}, {"network_interface": "a" * 16},
                        {"network_interface": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                setup.validate_profile(profile("slurm", **changes))
        for interface in (None, "ib0", "enp12s0f0", "bond0.200", "eth-test"):
            self.assertEqual(setup.validate_profile(profile("slurm", network_interface=interface))["network_interface"], interface)

    def test_loading_validates_json_and_rejects_duplicate_keys(self):
        self.path.write_text(json.dumps(profile()), encoding="utf-8")
        self.assertEqual(setup.load_profile(self.path), profile())
        for content in ('{"mode":"manual", "mode":"slurm"}', "[]", "not json"):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    setup.load_profile(self.path)
        with self.assertRaises(FileNotFoundError):
            setup.load_profile(self.directory / "missing.json")

    def test_manual_resolution_ignores_ray_environment_and_does_not_query_slurm(self):
        environment = {**self.allocation, "RAY_ADDRESS": "stale:1111", "RAY_HEAD_ADDRESS": "stale:2222",
                       "RAY_HEAD_PORT": "invalid"}
        with patch.dict(os.environ, environment, clear=True), patch.object(setup.subprocess, "check_output") as command:
            resolved = setup.resolve_profile(profile(head_port=6380))
        self.assertEqual(resolved.ray_init_address, "192.0.2.10:6380")
        command.assert_not_called()

    def test_slurm_resolution_requires_both_allocation_variables(self):
        for environment in ({}, {"SLURM_JOB_ID": "123"}, {"SLURM_JOB_NODELIST": "gpu01"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                    patch.object(setup.subprocess, "check_output") as command:
                with self.assertRaisesRegex(RuntimeError, "current allocation"):
                    setup.resolve_profile(profile("slurm"))
                command.assert_not_called()

    def test_slurm_resolution_uses_current_first_hostname_and_explicit_port(self):
        environment = {**self.allocation, "RAY_ADDRESS": "stale:1111", "RAY_HEAD_ADDRESS": "stale:2222",
                       "RAY_HEAD_PORT": "invalid"}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(setup.subprocess, "check_output", return_value="gpu01\ngpu02\ngpu03\n") as command:
            resolved = setup.resolve_profile(profile("slurm", head_port=6380))
        self.assertEqual(resolved.ray_init_address, "gpu01:6380")
        command.assert_called_once_with(["scontrol", "show", "hostnames", "gpu[01-03]"],
                                        text=True, stderr=subprocess.PIPE, timeout=5)

    def test_slurm_discovery_errors_fail_without_ray_auto_fallback(self):
        for failure in (FileNotFoundError("scontrol"), subprocess.TimeoutExpired("scontrol", 5),
                        subprocess.CalledProcessError(1, "scontrol")):
            with self.subTest(failure=failure), patch.dict(os.environ, self.allocation, clear=True), \
                    patch.object(setup.subprocess, "check_output", side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, "Cannot discover"):
                    setup.resolve_profile(profile("slurm"))
        for output in ("", "gpu01\ngpu01\n", "gpu01;echo unsafe\n", "gpu01:6379\n"):
            with self.subTest(output=output), patch.dict(os.environ, self.allocation, clear=True), \
                    patch.object(setup.subprocess, "check_output", return_value=output):
                with self.assertRaisesRegex(RuntimeError, "invalid allocation hostname"):
                    setup.resolve_profile(profile("slurm"))

    def test_slurm_interface_resolution_queries_only_allocated_head(self):
        with patch.dict(os.environ, self.allocation, clear=True), patch.object(
                setup.subprocess, "check_output", side_effect=["gpu01\ngpu02\n", interface_data("10.20.0.1")]) as command:
            resolved = setup.resolve_profile(profile("slurm", network_interface="ib0", head_port=6380))
        self.assertEqual(resolved.ray_init_address, "10.20.0.1:6380")
        self.assertEqual(command.call_args_list[1].args[0], [
            "srun", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--gres=none",
            "--nodelist", "gpu01", "ip", "-j", "-4", "address", "show", "dev", "ib0"])
        self.assertEqual(command.call_args_list[1].kwargs,
                         {"text": True, "stderr": subprocess.PIPE, "timeout": 15})

    def test_slurm_interface_must_have_one_global_usable_ipv4(self):
        invalid = [interface_data(), interface_data("10.20.0.1", "10.20.0.2"), interface_data("127.0.0.1"),
                   interface_data("169.254.0.1"), interface_data("224.0.0.1"), interface_data("0.0.0.0"),
                   interface_data(123), interface_data("not-ipv4"), "not-json", "{}", '[{"addr_info": null}]',
                   '[{"addr_info": [{"family": "inet", "scope": "link", "local": "10.20.0.1"}]}]']
        for output in invalid:
            with self.subTest(output=output), patch.dict(os.environ, self.allocation, clear=True), \
                    patch.object(setup.subprocess, "check_output", side_effect=["gpu01\n", output]):
                with self.assertRaises(RuntimeError):
                    setup.resolve_profile(profile("slurm", network_interface="ib0"))

    def test_slurm_interface_query_failures_propagate(self):
        for failure in (FileNotFoundError("srun"), subprocess.TimeoutExpired("srun", 15),
                        subprocess.CalledProcessError(1, "srun")):
            with self.subTest(failure=failure), patch.dict(os.environ, self.allocation, clear=True), \
                    patch.object(setup.subprocess, "check_output", side_effect=["gpu01\n", failure]):
                with self.assertRaisesRegex(RuntimeError, "Cannot query interface"):
                    setup.resolve_profile(profile("slurm", network_interface="ib0"))

    def test_manual_wizard_retries_bad_answers_and_prints_per_node_commands(self):
        answers = ["invalid", "manual", "head.example", "192.0.2.10", "192.0.2.10", "192.0.2.11, 192.0.2.12",
                   "0", "bad", "6380"]
        with patch.object(setup.subprocess, "check_output") as command:
            code, output = self.wizard(answers)
        self.assertEqual(code, 0)
        self.assertEqual(setup.load_profile(self.path), profile(head_port=6380, worker_addresses=["192.0.2.11", "192.0.2.12"]))
        self.assertIn("Invalid input:", output)
        self.assertIn("VLLM_HOST_IP=192.0.2.10 ray start --head --node-ip-address=192.0.2.10 --port=6380", output)
        self.assertIn("VLLM_HOST_IP=192.0.2.11 ray start --node-ip-address=192.0.2.11 --address=192.0.2.10:6380", output)
        self.assertIn("Run each command on its corresponding node", output)
        self.assertIn("--models MODEL", output)
        self.assertNotIn("--model-path", output)
        command.assert_not_called()

    def test_manual_wizard_accepts_single_node_and_default_port(self):
        code, output = self.wizard(["", "192.0.2.10", "", ""])
        self.assertEqual(code, 0)
        self.assertEqual(setup.load_profile(self.path), profile())
        self.assertIn("existing registered workers may still be used", output)

    def test_slurm_wizard_can_save_before_allocation_without_discovery(self):
        with patch.object(setup.subprocess, "check_output") as command:
            code, output = self.wizard(["slurm", "", "ib0"])
        self.assertEqual(code, 0)
        self.assertEqual(setup.load_profile(self.path), profile("slurm", network_interface="ib0"))
        self.assertIn("discovery is deferred", output)
        self.assertIn('scontrol show hostnames "$SLURM_JOB_NODELIST"', output)
        self.assertIn("Submit a new allocation with scripts/submit_jobs.sh", output)
        self.assertIn("scripts/slurm_head.sh on the first node inside an existing allocation", output)
        command.assert_not_called()

    def test_slurm_wizard_prints_allocated_roles_but_never_persists_hosts(self):
        with patch.object(setup.subprocess, "check_output", return_value="gpu01\ngpu02\ngpu03\n") as command:
            code, output = self.wizard(["2", "6380", ""], self.allocation)
        self.assertEqual(code, 0)
        self.assertIn("Current head: gpu01", output)
        self.assertIn("Current workers: gpu02, gpu03", output)
        self.assertEqual(setup.load_profile(self.path), profile("slurm", head_port=6380))
        self.assertNotIn("gpu01", self.path.read_text())
        self.assertEqual(command.call_count, 1)

    def test_slurm_wizard_preserves_reusable_profile_when_discovery_unavailable(self):
        with patch.object(setup.subprocess, "check_output", side_effect=FileNotFoundError("scontrol")):
            code, output = self.wizard(["2", "", ""], self.allocation)
        self.assertEqual(code, 0)
        self.assertIn("Allocation discovery unavailable", output)
        self.assertEqual(setup.load_profile(self.path), profile("slurm"))

    def test_eof_and_interrupt_leave_no_profile_or_temporary_file(self):
        answers = ["manual", "192.0.2.10", "", ""]
        for index in range(len(answers)):
            for error, expected in ((EOFError(), 1), (KeyboardInterrupt(), 130)):
                with self.subTest(index=index, error=type(error).__name__):
                    code, output = self.wizard(answers[:index] + [error])
                    self.assertEqual(code, expected)
                    self.assertIn("cancelled", output)
                    self.assertEqual(list(self.directory.iterdir()), [])

    def test_existing_output_is_never_overwritten_or_prompted(self):
        self.path.write_text("existing content", encoding="utf-8")
        with patch("builtins.input") as prompt, patch("sys.stdout", io.StringIO()):
            self.assertEqual(setup.configure(self.path), 1)
        prompt.assert_not_called()
        self.assertEqual(self.path.read_text(), "existing content")

    def test_concurrent_output_creation_is_preserved(self):
        real_link = os.link
        def publish(source, destination):
            Path(destination).write_text("created concurrently", encoding="utf-8")
            real_link(source, destination)
        with patch.object(setup.os, "link", side_effect=publish):
            code, output = self.wizard(["", "192.0.2.10", "", ""])
        self.assertEqual(code, 1)
        self.assertIn("Cannot create", output)
        self.assertEqual(self.path.read_text(), "created concurrently")
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_interrupted_or_failed_write_does_not_publish_partial_json(self):
        for failure, expected in ((KeyboardInterrupt(), 130), (OSError("disk full"), 1)):
            with self.subTest(failure=type(failure).__name__), patch.object(setup.json, "dump", side_effect=failure):
                code, _ = self.wizard(["", "192.0.2.10", "", ""])
            self.assertEqual(code, expected)
            self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

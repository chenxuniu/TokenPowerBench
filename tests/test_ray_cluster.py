"""Ray address precedence and SLURM discovery without starting a cluster."""

import os
import subprocess
import unittest
from unittest.mock import patch

from tokenpowerbench.distributed import RayClusterConfig


class RayClusterConfigTests(unittest.TestCase):
    def test_native_addresses_preserve_ports_and_support_ipv6(self):
        cases = {"head": "head:6379", "head:6380": "head:6380",
                 "127.0.0.1": "127.0.0.1:6379", "::1": "[::1]:6379",
                 "[2001:db8::1]:6380": "[2001:db8::1]:6380", "auto": "auto"}
        for address, expected in cases.items():
            with self.subTest(address=address):
                self.assertEqual(RayClusterConfig(head_address=address).ray_init_address, expected)

    def test_ray_client_and_worker_start_addresses_are_distinct(self):
        self.assertEqual(RayClusterConfig(head_address="ray://head:10002").ray_init_address,
                         "ray://head:10002")
        self.assertEqual(RayClusterConfig(head_address="ray://head").ray_init_address,
                         "ray://head:10001")
        self.assertEqual(RayClusterConfig(head_address="head", head_port=6380).ray_start_address, "head:6380")
        for address in ("auto", "local", "ray://head:10001"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                RayClusterConfig(head_address=address).ray_start_address

    def test_explicit_options_override_environment_and_slurm(self):
        env = {"RAY_HEAD_ADDRESS": "env-head:6380", "RAY_HEAD_PORT": "6381",
               "RAY_ADDRESS": "other-head:6382", "SLURM_JOB_NODELIST": "node[01-02]"}
        with patch.dict(os.environ, env, clear=True), patch("subprocess.check_output") as discovery:
            config = RayClusterConfig.resolve("cli-head:6383", 6384)
            self.assertEqual(config.ray_init_address, "cli-head:6384")
            self.assertEqual(RayClusterConfig.resolve(head_port=6385).ray_init_address, "env-head:6385")
        discovery.assert_not_called()

    def test_environment_override_beats_slurm_and_preserves_embedded_port(self):
        with patch.dict(os.environ, {"RAY_HEAD_ADDRESS": "env-head:6380", "SLURM_JOB_NODELIST": "nodes"}, clear=True), patch("subprocess.check_output") as discovery:
            self.assertEqual(RayClusterConfig.from_slurm().ray_init_address, "env-head:6380")
        discovery.assert_not_called()

    def test_ray_address_environment_is_supported(self):
        with patch.dict(os.environ, {"RAY_ADDRESS": "ray://head:10001"}, clear=True):
            self.assertEqual(RayClusterConfig.from_env().ray_init_address, "ray://head:10001")
            self.assertEqual(RayClusterConfig.resolve().ray_init_address, "ray://head:10001")

    def test_unused_environment_port_does_not_override_explicit_endpoint(self):
        with patch.dict(os.environ, {"RAY_HEAD_ADDRESS": "env-head:6380", "RAY_HEAD_PORT": "invalid"}, clear=True):
            self.assertEqual(RayClusterConfig.resolve("cli-head:6381").ray_init_address, "cli-head:6381")
            self.assertEqual(RayClusterConfig.from_env().ray_init_address, "env-head:6380")
            self.assertEqual(RayClusterConfig.resolve("auto").ray_init_address, "auto")
            with self.assertRaises(ValueError):
                RayClusterConfig.resolve("bare-head")

    def test_slurm_uses_first_hostname_and_bounded_command(self):
        env = {"SLURM_JOB_NODELIST": "gpu[01-02]", "RAY_HEAD_PORT": "6380"}
        with patch.dict(os.environ, env, clear=True), patch("subprocess.check_output", return_value="gpu01\ngpu02\n") as command:
            self.assertEqual(RayClusterConfig.resolve().ray_init_address, "gpu01:6380")
        command.assert_called_once_with(["scontrol", "show", "hostnames", "gpu[01-02]"],
                                        text=True, stderr=subprocess.DEVNULL, timeout=5)

    def test_slurm_errors_fall_back_to_existing_cluster_discovery(self):
        env = {"SLURM_JOB_NODELIST": "nodes"}
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("scontrol", 5)):
            with self.subTest(error=type(error).__name__), patch.dict(os.environ, env, clear=True), patch("subprocess.check_output", side_effect=error), self.assertWarns(RuntimeWarning):
                self.assertEqual(RayClusterConfig.resolve().ray_init_address, "auto")
        with patch.dict(os.environ, env, clear=True), patch("subprocess.check_output", return_value=""), self.assertWarns(RuntimeWarning):
            self.assertEqual(RayClusterConfig.resolve().ray_init_address, "auto")

    def test_no_environment_uses_discovery_without_commands(self):
        with patch.dict(os.environ, {}, clear=True), patch("subprocess.check_output") as command:
            self.assertEqual(RayClusterConfig.resolve().ray_init_address, "auto")
        command.assert_not_called()

    def test_invalid_addresses_and_ports_are_rejected(self):
        for address in ("", "two hosts", "host:noport", "host:0", "host:65536", "https://head:1234",
                        "[::1", "ray://user:secret@head:10001", "host/path", "ray://head:0"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                RayClusterConfig(head_address=address)
        for port in (0, 65536, True, 1.5, "noport"):
            with self.subTest(port=port), self.assertRaises(ValueError):
                RayClusterConfig(head_port=port)


if __name__ == "__main__":
    unittest.main()

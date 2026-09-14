"""Cluster setup and benchmark entry points share saved profile semantics."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_multi_node as runner


def profile(mode="manual", **overrides):
    return {"schema_version": 1, "mode": mode,
            "head_address": "10.0.0.1" if mode == "manual" else None,
            "head_port": 6381, "worker_addresses": ["10.0.0.2"] if mode == "manual" else [],
            "network_interface": None, **overrides}


class MultiNodeClusterConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cluster profile.json"

    def arguments(self, configuration=None, *extra):
        self.path.write_text(json.dumps(configuration or profile()))
        return runner.parse_args(["--models", "example-model", "--cluster-config", str(self.path), *extra])

    def test_setup_runs_without_model_or_inference_initialization(self):
        with patch("tokenpowerbench.distributed.cluster_setup.configure", return_value=0) as wizard, \
                patch.object(runner, "_run_suite") as benchmark:
            self.assertEqual(runner.run(["--configure-cluster", "--cluster-config", str(self.path)]), 0)
        wizard.assert_called_once_with(self.path.absolute())
        benchmark.assert_not_called()

    def test_setup_default_path_and_cancellation_status(self):
        with patch("tokenpowerbench.distributed.cluster_setup.configure", return_value=130) as wizard:
            self.assertEqual(runner.run(["--configure-cluster"]), 130)
        wizard.assert_called_once_with(Path("cluster.json").absolute())

    def test_setup_rejects_workload_arguments_instead_of_ignoring_them(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            runner.run(["--configure-cluster", "--models", "ignored-model"])
        self.assertEqual(error.exception.code, 2)

    def test_setup_preserves_existing_dangling_symlink(self):
        destination = self.path.parent / "missing-destination.json"
        self.path.symlink_to(destination)
        with redirect_stdout(io.StringIO()), patch("builtins.input") as prompt:
            self.assertEqual(runner.run(["--configure-cluster", "--cluster-config", str(self.path)]), 1)
        prompt.assert_not_called()
        self.assertTrue(self.path.is_symlink())
        self.assertFalse(destination.exists())

    def test_saved_manual_profile_overrides_stale_environment(self):
        with patch.dict(os.environ, {"RAY_HEAD_ADDRESS": "wrong-host:7777", "RAY_HEAD_PORT": "9999"}, clear=True):
            args = self.arguments()
            self.assertEqual(runner._resolve_cluster(args).ray_init_address, "10.0.0.1:6381")
        self.assertEqual(args.cluster_profile, profile())
        self.assertEqual(args.cluster_config, self.path.resolve())

    def test_cli_address_and_port_override_profile_independently(self):
        cases = [(("--ray-head-address", "10.0.0.9"), "10.0.0.9:6381"),
                 (("--ray-head-address", "10.0.0.9:6382"), "10.0.0.9:6382"),
                 (("--ray-head-port", "6383"), "10.0.0.1:6383"),
                 (("--ray-head-address", "10.0.0.9:6382", "--ray-head-port", "6383"), "10.0.0.9:6383")]
        for options, expected in cases:
            with self.subTest(options=options):
                self.assertEqual(runner._resolve_cluster(self.arguments(None, *options)).ray_init_address, expected)

    def test_explicit_cli_head_can_override_slurm_profile_outside_allocation(self):
        args = self.arguments(profile("slurm"), "--ray-head-address", "10.0.0.9")
        with patch.dict(os.environ, {}, clear=True), patch("subprocess.check_output") as discovery:
            self.assertEqual(runner._resolve_cluster(args).ray_init_address, "10.0.0.9:6381")
        discovery.assert_not_called()

    def test_slurm_profile_resolves_each_current_allocation(self):
        args = self.arguments(profile("slurm"))
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_JOB_NODELIST": "gpu[01-02]",
                                    "RAY_ADDRESS": "stale-host:9999"}, clear=True), \
                patch("subprocess.check_output", side_effect=["gpu01\ngpu02\n", "gpu03\ngpu04\n"]):
            self.assertEqual(runner._resolve_cluster(args).ray_init_address, "gpu01:6381")
            self.assertEqual(runner._resolve_cluster(args).ray_init_address, "gpu03:6381")
        self.assertIsNone(args.cluster_profile["head_address"])

    def test_profile_errors_fail_preflight_before_engine_creation(self):
        self.path.write_text('{"mode":"manual","head_address":"10.0.0.1"}')
        with patch.object(runner, "VLLMDistributedEngine") as engine, redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            runner.run(["--models", "example", "--cluster-config", str(self.path)])
        self.assertEqual(error.exception.code, 2)
        engine.assert_not_called()

    def test_no_profile_preserves_environment_resolution(self):
        with patch.dict(os.environ, {"RAY_ADDRESS": "env-head:6385"}, clear=True):
            args = runner.parse_args(["--models", "example"])
            self.assertIsNone(args.cluster_profile)
            self.assertEqual(runner._resolve_cluster(args).ray_init_address, "env-head:6385")

    def test_artifacts_save_profile_and_effective_address(self):
        args = self.arguments(None, "--output-dir", self.directory.name, "--tensor-parallel", "1",
                              "--pipeline-parallel", "1", "--batch-sizes", "1")
        result = {"results": [], "performance_metrics": {"total_prompts": 1, "total_tokens": 1}}
        with patch.object(runner, "_load_workloads", return_value={"test": ["prompt"]}), \
                patch.object(runner, "_run_configuration", return_value=result), redirect_stdout(io.StringIO()):
            self.assertEqual(runner._run_suite(args), 0)
        saved = json.loads(next(Path(self.directory.name).glob("multi_*/config.json")).read_text())
        self.assertEqual(saved["cluster_profile"], profile())
        self.assertEqual(saved["ray_address"], "10.0.0.1:6381")


if __name__ == "__main__":
    unittest.main()

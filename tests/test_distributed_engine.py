"""Distributed orchestration tests without Ray services or GPU dependencies."""

import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tokenpowerbench.distributed import vllm_distributed as distributed
from tokenpowerbench.distributed.ray_cluster import RayClusterConfig


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, duration):
        self.value += duration


class Reference:
    def __init__(self, kind, value=None):
        self.kind = kind
        self.value = value


class RemoteMethod:
    def __init__(self, function):
        self.remote = function


class FakeActor:
    def __init__(self, ray, index, options, constructor):
        self.ray = ray
        self.index = index
        self.options = options
        self.constructor = constructor
        self.calls = []
        self.ready = RemoteMethod(lambda: Reference("ready", {"worker": index}))
        self.warmup = RemoteMethod(self._warmup)
        self.close = RemoteMethod(self._close)
        self.__call__ = RemoteMethod(self._call)

    def _warmup(self):
        self.ray.warmed.append(self)
        return Reference("warmup", {"output_tokens": [1]})

    def _close(self):
        self.ray.closed.append(self)
        return Reference("close")

    def _call(self, batch):
        self.calls.append(copy.deepcopy(batch))
        row_count = len(batch["text"])
        payload = {
            "request_id": batch["request_id"], "prompt": batch["text"],
            "generated_text": ["one word" for _ in range(row_count)],
            "input_tokens": [10 for _ in range(row_count)],
            "output_tokens": [7 for _ in range(row_count)],
            "batch_id": [f"worker-{self.index}-batch-{len(self.calls)}" for _ in range(row_count)],
            "batch_duration_s": [1.5 for _ in range(row_count)],
            "batch_size": [row_count for _ in range(row_count)],
        }
        if self.ray.corrupt is not None:
            payload = self.ray.corrupt(payload)
        return Reference("batch", payload)


class FakeActorType:
    def __init__(self, ray):
        self.ray = ray

    def options(self, **options):
        class Configured:
            def remote(inner, **constructor):
                if self.ray.actor_creation_error is not None and self.ray.actors:
                    raise self.ray.actor_creation_error
                self.ray.clock.advance(20)
                actor = FakeActor(self.ray, len(self.ray.actors), options, constructor)
                self.ray.actors.append(actor)
                return actor
        return Configured()


class FakeGroup:
    def __init__(self, bundles, strategy):
        self.bundles = bundles
        self.strategy = strategy
        self.state = "CREATED"

    def ready(self):
        return Reference("placement")


class FakeRay:
    __version__ = "2.48.0"

    def __init__(self, clock, initialized=False):
        self.clock = clock
        self.initialized = initialized
        self.actors = []
        self.groups = []
        self.warmed = []
        self.closed = []
        self.killed = []
        self.removed = []
        self.get_calls = []
        self.errors = {}
        self.wait_times_out = False
        self.actor_creation_error = None
        self.corrupt = None
        self.init = Mock(side_effect=self._init)
        self.shutdown = Mock(side_effect=self._shutdown)
        self.get_runtime_context = Mock(return_value=SimpleNamespace(gcs_address="192.0.2.10:6379"))
        self.nodes = Mock(return_value=[
            {"Alive": True, "Resources": {"CPU": 8}},
            {"Alive": True, "Resources": {"CPU": 8, "GPU": 8}},
        ])
        self.util = SimpleNamespace(placement_group=self._group, remove_placement_group=self._remove_group,
                                    placement_group_table=lambda group: {"state": group.state})
        self.remote_classes = []

    def is_initialized(self):
        return self.initialized

    def _init(self, **kwargs):
        self.initialized = True

    def _shutdown(self):
        self.initialized = False

    def remote(self, actor_class):
        self.remote_classes.append(actor_class)
        return FakeActorType(self)

    def _group(self, bundles, strategy):
        group = FakeGroup(bundles, strategy)
        self.groups.append(group)
        return group

    def _remove_group(self, group):
        self.removed.append(group)
        group.state = "REMOVED"

    def get(self, reference, timeout):
        self.get_calls.append((reference, timeout))
        if isinstance(reference, list):
            return [self.get(item, timeout) for item in reference]
        if reference.kind in self.errors:
            raise self.errors[reference.kind]
        if reference.kind in ("ready", "warmup"):
            self.clock.advance(3)
        return copy.deepcopy(reference.value)

    def wait(self, pending, num_returns, timeout):
        if self.wait_times_out:
            return [], pending
        self.clock.advance(2)
        # Deliberately finish later-submitted work first to exercise ordering.
        return [pending[-1]], pending[:-1]

    def kill(self, actor, no_restart):
        assert no_restart is True
        self.killed.append(actor)


def config(**overrides):
    return {"model_path": "test-model", "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2, "concurrency": 2, "batch_size": 2, **overrides}


class DistributedEngineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.ray = FakeRay(self.clock)
        real_import = distributed.importlib.import_module
        def import_module(name):
            if name == "ray":
                return self.ray
            if name == "ray.util.scheduling_strategies":
                return SimpleNamespace(PlacementGroupSchedulingStrategy=lambda **kwargs: SimpleNamespace(**kwargs))
            return real_import(name)
        for active_patch in (patch.object(distributed.importlib, "import_module", side_effect=import_module),
                             patch.object(distributed.time, "perf_counter", self.clock),
                             patch.object(distributed.time, "sleep", side_effect=self.clock.advance),
                             patch.dict(sys.modules, {
                                 "packaging": SimpleNamespace(),
                                 "packaging.version": SimpleNamespace(Version=lambda value: tuple(map(int, value.split(".")))),
                             })):
            active_patch.start()
            self.addCleanup(active_patch.stop)

    def engine(self, **overrides):
        return distributed.VLLMDistributedEngine(RayClusterConfig(head_address="auto"), config(**overrides))

    def test_prepare_allocates_exact_bundles_pins_coordinator_and_warms_every_actor(self):
        engine = self.engine()
        predictor = type("StubPredictor", (), {})
        with patch.object(distributed, "VLLMPredictor", predictor):
            info = engine.prepare()
        self.assertEqual(self.ray.remote_classes, [predictor])
        self.assertEqual(len(self.ray.groups), 2)
        self.assertEqual([len(group.bundles) for group in self.ray.groups], [4, 4])
        self.assertEqual(sum(bundle["GPU"] for group in self.ray.groups for bundle in group.bundles), 8)
        self.assertEqual(self.ray.warmed, self.ray.actors)
        for actor in self.ray.actors:
            options = actor.options
            self.assertEqual((options["num_gpus"], options["num_cpus"]), (0, 1))
            self.assertEqual(options["scheduling_strategy"].placement_group_bundle_index, 0)
            self.assertTrue(options["scheduling_strategy"].placement_group_capture_child_tasks)
            self.assertEqual(options["max_restarts"], 0)
            self.assertEqual(options["max_task_retries"], 0)
            self.assertNotIn("stop", actor.constructor["sampling_params"])
            self.assertEqual(actor.constructor["sampling_params"]["seed"], 42)
        self.assertGreater(info["duration_s"], 40)
        self.assertEqual(engine.prepare(), info)
        self.assertEqual(len(self.ray.warmed), 2)
        engine.close()

    def test_batch_dispatch_exact_tokens_order_and_unique_batch_duration(self):
        engine = self.engine()
        result = engine.run_benchmark(["same", "same", "third", "fourth", "last"])
        metrics = result["performance_metrics"]
        self.assertEqual(metrics["total_tokens"], 35)  # Text contains only two words per response.
        self.assertEqual(metrics["total_prompts"], 5)
        self.assertEqual(metrics["total_batches"], 3)
        self.assertEqual(metrics["sum_worker_batch_time_s"], 4.5)
        self.assertEqual(metrics["avg_batch_time_s"], 1.5)
        self.assertIsNone(metrics["avg_processing_time_s"])
        self.assertEqual([row["request_id"] for row in result["results"]], list(range(5)))
        self.assertTrue(all(row["processing_time_s"] is None for row in result["results"]))
        self.assertEqual(sorted(len(batch["text"]) for actor in self.ray.actors for batch in actor.calls), [1, 2, 2])
        window = result["measurement_window"]
        self.assertEqual(window["end_s"] - window["start_s"], 6)
        self.assertEqual(metrics["total_time_s"], 6)
        self.assertEqual(window["scope"], "driver_dispatch_and_collection")
        self.assertEqual(window["clock"], "time.perf_counter")
        self.assertGreater(result["preparation"]["duration_s"], metrics["total_time_s"])
        self.assertEqual(self.ray.killed, self.ray.actors)
        self.assertEqual(self.ray.removed, self.ray.groups)
        self.ray.shutdown.assert_called_once()
        json.dumps(result, allow_nan=False)

    def test_explicit_prepare_reuses_workers_until_close_and_preserves_existing_ray(self):
        self.ray.initialized = True
        engine = self.engine()
        engine.prepare()
        for _ in range(2):
            engine.run_benchmark(["first", "second"])
        self.assertEqual(len(self.ray.actors), 2)
        self.assertEqual(len(self.ray.warmed), 2)
        self.assertEqual(self.ray.closed, [])
        self.ray.init.assert_not_called()
        engine.close()
        engine.close()
        self.ray.shutdown.assert_not_called()
        self.assertEqual(len(self.ray.killed), 2)
        self.assertEqual(len(self.ray.removed), 2)

    def test_standalone_repeated_runs_create_fresh_owned_resources(self):
        engine = self.engine()
        engine.run_benchmark(["a"])
        engine.run_benchmark(["b"])
        self.assertEqual(len(self.ray.actors), 4)
        self.assertEqual(len(self.ray.killed), 4)
        self.assertEqual(self.ray.init.call_count, 2)
        self.assertEqual(self.ray.shutdown.call_count, 2)

    def test_existing_explicit_cluster_matches_and_records_actual_address(self):
        self.ray.initialized = True
        engine = distributed.VLLMDistributedEngine(RayClusterConfig("192.0.2.10:6379"), config())
        info = engine.prepare()["cluster"]
        self.assertEqual(info["requested_address"], "192.0.2.10:6379")
        self.assertEqual(info["gcs_address"], "192.0.2.10:6379")
        engine.close()
        self.ray.init.assert_not_called()
        self.ray.shutdown.assert_not_called()

    def test_existing_cluster_accepts_hostname_alias_and_equivalent_ipv6(self):
        self.ray.initialized = True
        cases = [("HEAD.example.:6379", "192.0.2.10:6379"),
                 ("[2001:db8:0:0::1]:6379", "[2001:db8::1]:6379"),
                 ("[::ffff:192.0.2.10]:6379", "192.0.2.10:6379")]
        for requested, actual in cases:
            with self.subTest(requested=requested):
                self.ray.get_runtime_context.return_value.gcs_address = actual
                with patch.object(distributed.socket, "getaddrinfo", return_value=[
                        (2, 1, 6, "", ("192.0.2.10", 0))]):
                    engine = distributed.VLLMDistributedEngine(RayClusterConfig(requested), config())
                    engine.prepare()
                    engine.close()
        self.ray.shutdown.assert_not_called()

    def test_existing_other_cluster_is_rejected_without_reservations_or_disconnect(self):
        self.ray.initialized = True
        for requested in ("192.0.2.11:6379", "192.0.2.10:6380", "ray://192.0.2.10:10001"):
            with self.subTest(requested=requested):
                engine = distributed.VLLMDistributedEngine(RayClusterConfig(requested), config())
                with self.assertRaisesRegex(RuntimeError, "Cannot verify.*existing Ray connection"):
                    engine.prepare()
                self.assertEqual(self.ray.groups, [])
                self.assertEqual(self.ray.actors, [])
                self.assertTrue(self.ray.is_initialized())
        self.ray.init.assert_not_called()
        self.ray.shutdown.assert_not_called()

    def test_existing_cluster_requires_verifiable_address_for_explicit_reuse(self):
        self.ray.initialized = True
        self.ray.get_runtime_context.side_effect = AttributeError("GCS address unavailable")
        engine = distributed.VLLMDistributedEngine(RayClusterConfig("192.0.2.10:6379"), config())
        with self.assertRaisesRegex(RuntimeError, "GCS address unavailable"):
            engine.prepare()
        self.assertEqual(self.ray.groups, [])
        automatic = self.engine()
        self.assertIsNone(automatic.prepare()["cluster"]["gcs_address"])
        automatic.close()
        self.ray.shutdown.assert_not_called()

    def test_local_requires_fresh_session_but_auto_can_reuse_existing_session(self):
        engine = distributed.VLLMDistributedEngine(RayClusterConfig("local"), config())
        engine.prepare()
        self.ray.init.assert_called_once_with(address="local", ignore_reinit_error=True)
        engine.close()
        self.ray.init.reset_mock()
        self.ray.shutdown.reset_mock()
        self.ray.initialized = True
        with self.assertRaisesRegex(RuntimeError, "requires a new instance"):
            engine.prepare()
        self.ray.init.assert_not_called()
        self.ray.shutdown.assert_not_called()

    def test_unresolvable_hostname_does_not_silently_reuse_existing_cluster(self):
        self.ray.initialized = True
        with patch.object(distributed.socket, "getaddrinfo", side_effect=OSError("DNS unavailable")):
            engine = distributed.VLLMDistributedEngine(RayClusterConfig("head.example:6379"), config())
            with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
                engine.prepare()
        self.assertEqual(self.ray.groups, [])
        self.ray.shutdown.assert_not_called()

    def test_preparation_failure_cleans_partially_created_actors_and_all_groups(self):
        failure = RuntimeError("actor construction failed")
        self.ray.actor_creation_error = failure
        engine = self.engine()
        with self.assertRaises(RuntimeError) as caught:
            engine.prepare()
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(self.ray.actors), 1)
        self.assertEqual(self.ray.killed, self.ray.actors)
        self.assertEqual(self.ray.removed, self.ray.groups)
        self.ray.shutdown.assert_called_once()

    def test_placement_timeout_removes_reservations(self):
        failure = TimeoutError("placement unavailable")
        self.ray.errors["placement"] = failure
        engine = self.engine(placement_timeout_s=4)
        with self.assertRaises(TimeoutError) as caught:
            engine.prepare()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.ray.actors, [])
        self.assertEqual(self.ray.removed, self.ray.groups)
        self.assertTrue(any(timeout == 4 for _, timeout in self.ray.get_calls))

    def test_inference_timeout_and_interrupt_release_workers_and_preserve_primary(self):
        for failure in (TimeoutError("batch timeout"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                self.ray.errors["batch"] = failure
                self.ray.errors["close"] = RuntimeError("shutdown also failed")
                engine = self.engine()
                with self.assertRaises(type(failure)) as caught:
                    engine.run_benchmark(["prompt"])
                self.assertIs(caught.exception, failure)
                self.assertTrue(engine.cleanup_errors)
                self.assertEqual(len(self.ray.killed), len(self.ray.actors))
                self.assertEqual(len(self.ray.removed), len(self.ray.groups))

    def test_wait_timeout_closes_explicitly_prepared_engine(self):
        engine = self.engine()
        engine.prepare()
        self.ray.wait_times_out = True
        with self.assertRaisesRegex(TimeoutError, "inference"):
            engine.run_benchmark(["prompt"])
        self.assertEqual(self.ray.killed, self.ray.actors)

    def test_capacity_checks_live_colocated_cpu_gpu_bundles(self):
        self.ray.nodes.return_value = [
            {"Alive": True, "Resources": {"GPU": 8, "CPU": 1}},
            {"Alive": True, "Resources": {"CPU": 100}},
            {"Alive": False, "Resources": {"GPU": 32, "CPU": 32}},
        ]
        with self.assertRaisesRegex(RuntimeError, "have 1"):
            self.engine().prepare()
        self.assertEqual(self.ray.groups, [])
        self.ray.shutdown.assert_called_once()

    def test_ray_query_failure_propagates_without_unbound_variables(self):
        failure = RuntimeError("cluster query unavailable")
        self.ray.nodes.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            self.engine().prepare()
        self.assertIs(caught.exception, failure)
        self.ray.shutdown.assert_called_once()

    def test_invalid_requests_and_configuration_fail_before_ray(self):
        for prompts in ("not a prompt list", [], [""], [None]):
            with self.subTest(prompts=prompts), self.assertRaises((TypeError, ValueError)):
                self.engine().run_benchmark(prompts)
        for overrides in ({"tensor_parallel_size": 0}, {"batch_size": True}, {"concurrency": 1.5},
                          {"max_tokens": 0}, {"seed": -1}, {"top_p": 2}, {"temperature": float("nan")},
                          {"inference_timeout_s": 0}, {"model_kwargs": []}):
            with self.subTest(overrides=overrides), self.assertRaises((TypeError, ValueError)):
                self.engine(**overrides)
        self.ray.init.assert_not_called()

    def test_missing_duplicate_or_malformed_output_is_never_success(self):
        corruptions = [
            lambda payload: {**payload, "request_id": [0, 0]},
            lambda payload: {**payload, "output_tokens": [7]},
            lambda payload: {key: value for key, value in payload.items() if key != "output_tokens"},
            lambda payload: {**payload, "prompt": ["wrong", "wrong"]},
            lambda payload: {**payload, "batch_size": [1, 1]},
            lambda payload: {**payload, "batch_duration_s": [1.0, 2.0]},
            lambda payload: {key: value[:1] for key, value in payload.items()},
        ]
        for corrupt in corruptions:
            with self.subTest(corruption=corrupt):
                self.ray.corrupt = corrupt
                with self.assertRaises((RuntimeError, ValueError)):
                    self.engine().run_benchmark(["first", "second"])
                self.assertEqual(len(self.ray.killed), len(self.ray.actors))

    def test_configuration_is_copied_and_v1_ray_requirement_is_checked(self):
        settings = config(model_kwargs={"dtype": "bfloat16"})
        engine = distributed.VLLMDistributedEngine(RayClusterConfig(), settings)
        settings["model_kwargs"]["dtype"] = "float32"
        self.assertEqual(engine.config["model_kwargs"]["dtype"], "bfloat16")
        self.ray.__version__ = "2.22.0"
        with self.assertRaisesRegex(RuntimeError, "2.43.0"):
            engine.prepare()
        self.ray.init.assert_not_called()


if __name__ == "__main__":
    unittest.main()

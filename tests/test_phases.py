"""CPU-only checks of phase boundaries, rejection paths, and token accounting."""

import subprocess
import sys
import types
import unittest
from unittest import mock

from tokenpowerbench.engines.vllm_engine import VLLMEngine
from tokenpowerbench.phases import PhaseTimingError, build_phase_event


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeStepEngine:
    """Incremental backend with explicitly controlled host times and tokens."""

    def __init__(self, clock, profiles):
        self.clock = clock
        self.profiles = iter(profiles)
        self.active = False
        self.aborted = []
        self.submissions = []

    def add_request(self, request_id, prompt, params):
        if self.active:
            raise AssertionError("requests overlapped")
        self.submissions.append((request_id, prompt))
        self.request_id = request_id
        self.steps = iter(next(self.profiles))
        self.active = True
        self.clock.now += 0.2

    def has_unfinished_requests(self):
        return self.active

    def step(self):
        delta, tokens, finished = next(self.steps)
        self.clock.now += delta
        if isinstance(tokens, Exception):
            raise tokens
        self.active = not finished
        if tokens is None:
            return []
        return [types.SimpleNamespace(
            request_id=self.request_id,
            outputs=[types.SimpleNamespace(token_ids=tokens, text="text length is irrelevant")],
            prompt_token_ids=[10, 11, 12],
            finished=finished,
        )]

    def abort_request(self, request_ids):
        if not isinstance(request_ids, list):
            raise AssertionError("vLLM V1 abort_request requires a list of request IDs")
        self.aborted.append(request_ids)
        self.active = False


class PhaseProfilingTests(unittest.TestCase):
    def make_engine(self, profiles):
        clock = Clock()
        backend = FakeStepEngine(clock, profiles)
        engine = VLLMEngine()
        engine._llm = types.SimpleNamespace(llm_engine=backend)
        engine._phase_profiling = True
        engine._sampling_params = mock.Mock(return_value=object())
        return clock, backend, engine

    def run_profile(self, engine, clock, count=1):
        with mock.patch("tokenpowerbench.engines.vllm_engine.time.perf_counter", clock):
            return engine.run_profiled_benchmark(["prompt"], count, 1, 16)

    def test_first_token_is_observed_before_full_completion(self):
        clock, backend, engine = self.make_engine([[
            (0.3, None, False),
            (0.5, [20], False),
            (0.8, [20, 21], True),
        ]])
        outputs, started, finished, events = self.run_profile(engine, clock)
        event = events[0]
        self.assertEqual((started, finished), (0.0, 1.8))
        self.assertAlmostEqual(event["ttft_s"], 1.0)
        self.assertAlmostEqual(event["prefill_proxy_s"], 1.0)
        self.assertEqual(event["prefill_start_s"], event["submitted_s"])
        self.assertEqual(event["dispatch_completed_s"], 0.2)
        self.assertNotIn("execution_start_s", event)
        self.assertAlmostEqual(event["decode_s"], 0.8)
        self.assertEqual(event["input_tokens"], 3)
        self.assertEqual(event["output_tokens"], 2)
        self.assertEqual(engine.estimate_tokens(outputs), 2)
        self.assertLess(event["first_token_s"], event["finished_s"])
        self.assertEqual(event["timing_source"], "host_engine_step")
        self.assertEqual(backend.aborted, [])

    def test_prefill_includes_dispatch_when_background_execution_starts_early(self):
        clock, backend, engine = self.make_engine([[
            (0.1, [20], False), (0.4, [20, 21], True),
        ]])
        _, _, _, events = self.run_profile(engine, clock)
        event = events[0]
        # Dispatch takes 0.2 s, during which an asynchronous engine may execute.
        # It must remain inside the energy window, even when step returns fast.
        self.assertAlmostEqual(event["prefill_proxy_s"], 0.3)
        self.assertEqual(event["prefill_proxy_s"], event["ttft_s"])
        self.assertLess(event["prefill_start_s"], event["dispatch_completed_s"])
        self.assertAlmostEqual(event["prefill_proxy_s"] + event["decode_s"],
                               event["request_latency_s"])

    def test_serial_requests_have_distinct_ids_and_nonoverlapping_intervals(self):
        clock, backend, engine = self.make_engine([
            [(0.5, [20], False), (0.4, [20, 21], True)],
            [(0.6, [30], True)],
        ])
        _, _, _, events = self.run_profile(engine, clock, count=2)
        self.assertLessEqual(events[0]["finished_s"], events[1]["submitted_s"])
        self.assertNotEqual(events[0]["request_id"], events[1]["request_id"])
        self.assertEqual(events[1]["decode_s"], 0.0)
        self.assertEqual(len(backend.submissions), 2)

    def test_coalesced_first_observation_is_rejected_and_aborted(self):
        clock, backend, engine = self.make_engine([[(1.0, [20, 21], True)]])
        with self.assertRaisesRegex(PhaseTimingError, "multiple tokens"):
            self.run_profile(engine, clock)
        self.assertEqual(backend.aborted, [[backend.submissions[0][0]]])

    def test_no_completed_output_is_not_reported_as_success(self):
        clock, backend, engine = self.make_engine([[(1.0, None, True)]])
        with self.assertRaisesRegex(PhaseTimingError, "without a completed"):
            self.run_profile(engine, clock)
        self.assertEqual(len(backend.aborted), 1)

    def test_empty_final_tokens_are_rejected(self):
        clock, backend, engine = self.make_engine([[(1.0, [], True)]])
        with self.assertRaisesRegex(PhaseTimingError, "without a first token"):
            self.run_profile(engine, clock)
        self.assertEqual(len(backend.aborted), 1)

    def test_noncumulative_output_is_rejected(self):
        clock, backend, engine = self.make_engine([[
            (0.5, [20], False), (0.3, [21], True),
        ]])
        with self.assertRaisesRegex(PhaseTimingError, "not cumulative"):
            self.run_profile(engine, clock)
        self.assertEqual(len(backend.aborted), 1)

    def test_backend_failure_propagates_and_aborts(self):
        clock, backend, engine = self.make_engine([[(0.5, RuntimeError("GPU failed"), False)]])
        with self.assertRaisesRegex(RuntimeError, "GPU failed"):
            self.run_profile(engine, clock)
        self.assertEqual(len(backend.aborted), 1)

    def test_requires_serial_batch_and_phase_model_settings(self):
        clock, backend, engine = self.make_engine([])
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            engine.run_profiled_benchmark(["prompt"], 1, 2, 16)
        engine._phase_profiling = False
        with self.assertRaisesRegex(PhaseTimingError, "phase_profiling=True"):
            self.run_profile(engine, clock)
        self.assertEqual(backend.submissions, [])

    def test_active_engine_is_rejected_before_submitting(self):
        clock, backend, engine = self.make_engine([])
        backend.active = True
        with self.assertRaisesRegex(PhaseTimingError, "idle engine"):
            self.run_profile(engine, clock)
        self.assertEqual(backend.submissions, [])
        self.assertEqual(backend.aborted, [])


class EventAndEngineTests(unittest.TestCase):
    def test_event_rejects_reversed_or_missing_boundaries(self):
        fields = dict(
            request_id="r", submitted_s=1.0, dispatch_completed_s=2.0,
            first_token_s=3.0, finished_s=4.0, input_tokens=10, output_tokens=2,
        )
        for changes in ({"first_token_s": 1.0}, {"dispatch_completed_s": 0.0},
                        {"dispatch_completed_s": 3.5}, {"finished_s": float("nan")},
                        {"output_tokens": 0}):
            with self.subTest(changes=changes), self.assertRaises(PhaseTimingError):
                build_phase_event(**(fields | changes))

    def test_count_uses_engine_token_ids_instead_of_text(self):
        engine = VLLMEngine()
        output = types.SimpleNamespace(outputs=[
            types.SimpleNamespace(token_ids=[1, 2, 3, 4], text="singleword"),
        ])
        self.assertEqual(engine.estimate_tokens([output]), 4)
        output.outputs[0] = types.SimpleNamespace(text="many words cannot prove token count")
        with self.assertRaisesRegex(ValueError, "no generated token IDs"):
            engine.estimate_tokens([output])

    def test_batch_workload_repeats_exactly_and_handles_partial_batch(self):
        engine = VLLMEngine()
        engine._llm = object()
        calls = []

        def inference(prompts, batch_size, max_tokens):
            calls.append(prompts)
            return [object() for _ in prompts]

        engine.run_inference = inference
        outputs, _, _ = engine.run_benchmark(["a", "b"], 3, 2, 4)
        self.assertEqual(calls, [["a", "b"], ["a"]])
        self.assertEqual(len(outputs), 3)

    def test_invalid_workload_fails_without_loading_cuda(self):
        engine = VLLMEngine()
        for args in (([], 1, 1, 1), (["p"], 0, 1, 1), (["p"], 1, 0, 1), (["p"], 1, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                engine.run_benchmark(*args)

    def test_phase_setup_disables_caches_and_overlapping_sequences(self):
        fake_llm = mock.Mock(return_value=object())
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            empty_cache=mock.Mock(), device_count=mock.Mock(return_value=2),
        ))
        fake_vllm = types.SimpleNamespace(LLM=fake_llm)
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "vllm": fake_vllm}), \
             mock.patch.object(VLLMEngine, "available", new_callable=mock.PropertyMock, return_value=True):
            engine = VLLMEngine()
            engine.setup_model("model", phase_profiling=True, seed=42, temperature=0)
        kwargs = fake_llm.call_args.kwargs
        self.assertFalse(kwargs["enable_prefix_caching"])
        self.assertFalse(kwargs["enable_chunked_prefill"])
        self.assertEqual(kwargs["max_num_seqs"], 1)
        self.assertEqual(kwargs["tensor_parallel_size"], 2)
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(engine._temperature, 0)
        self.assertTrue(kwargs["trust_remote_code"])
        self.assertEqual(engine.resolved_config["requested"], kwargs)
        self.assertIsNone(engine.resolved_config["effective"]["enable_chunked_prefill"])

    def test_effective_engine_settings_are_recorded_when_vllm_overrides_requests(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            empty_cache=mock.Mock(), device_count=mock.Mock(return_value=1),
        ))
        config = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(enable_chunked_prefill=True,
                                                   max_num_seqs=1, max_num_batched_tokens=4096),
            cache_config=types.SimpleNamespace(enable_prefix_caching=False,
                                               gpu_memory_utilization=0.8),
            model_config=types.SimpleNamespace(max_model_len=4096),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        )
        fake_llm = types.SimpleNamespace(llm_engine=types.SimpleNamespace(vllm_config=config))
        fake_vllm = types.SimpleNamespace(LLM=mock.Mock(return_value=fake_llm))
        for profile in (False, True):
            with self.subTest(phase_profiling=profile), \
                 mock.patch.dict(sys.modules, {"torch": fake_torch, "vllm": fake_vllm}), \
                 mock.patch.object(VLLMEngine, "available", new_callable=mock.PropertyMock, return_value=True):
                engine = VLLMEngine()
                engine.setup_model("model", phase_profiling=profile, max_model_len=4096,
                                   gpu_memory_utilization=0.8)
            self.assertTrue(engine.resolved_config["effective"]["enable_chunked_prefill"])
            self.assertFalse(engine.resolved_config["effective"]["enable_prefix_caching"])
            self.assertEqual(engine.resolved_config["effective"]["max_model_len"], 4096)
            self.assertEqual(engine.resolved_config["effective"]["max_num_batched_tokens"], 4096)
            if profile:
                self.assertFalse(engine.resolved_config["requested"]["enable_chunked_prefill"])

    def test_import_does_not_require_torch_or_vllm(self):
        from pathlib import Path
        import tokenpowerbench

        package_root = str(Path(tokenpowerbench.__file__).resolve().parent.parent)
        result = subprocess.run([
            sys.executable, "-S", "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from tokenpowerbench.engines import VLLMEngine; "
            "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules",
            package_root,
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

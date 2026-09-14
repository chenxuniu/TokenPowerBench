"""Distributed worker behavior tested without Ray, vLLM, or GPU hardware."""

import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tokenpowerbench.distributed.predictor import VLLMPredictor


def response(prompt, tokens=(10, 11), text="one", finished=True):
    return SimpleNamespace(prompt=prompt, prompt_token_ids=[1, 2, 3], finished=finished,
                           outputs=[SimpleNamespace(token_ids=list(tokens), text=text)])


class SamplingParams:
    def __init__(self, **kwargs):
        self.n = 1
        self.seed = None
        self.__dict__.update(kwargs)


class DistributedPredictorTests(unittest.TestCase):
    def make_predictor(self, **kwargs):
        shutdown = Mock()
        llm = SimpleNamespace(generate=Mock(),
                              llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)))
        module = SimpleNamespace(LLM=Mock(return_value=llm), SamplingParams=SamplingParams)
        active = patch.dict(sys.modules, {"vllm": module})
        active.start()
        self.addCleanup(active.stop)
        predictor = VLLMPredictor(model_path=kwargs.pop("model_path", "model"),
                                  tensor_parallel_size=2, pipeline_parallel_size=2,
                                  sampling_params=kwargs.pop("sampling_params", {"n": 1, "max_tokens": 32, "seed": 42}),
                                  **kwargs)
        self.addCleanup(predictor.close)
        return predictor, llm, module

    def invoke(self, predictor, prompts=("first", "second")):
        with patch("tokenpowerbench.distributed.predictor.time.perf_counter", side_effect=[10.0, 12.5]):
            return predictor({"request_id": list(range(len(prompts))), "text": list(prompts)})

    def test_one_generation_call_per_batch_with_raw_prompts_and_exact_tokens(self):
        predictor, llm, module = self.make_predictor()
        llm.generate.return_value = [response("first", (10, 11, 12), "one"), response("second", (20,), "")]
        result = self.invoke(predictor)
        llm.generate.assert_called_once_with(["first", "second"], predictor.sampling_params, use_tqdm=False)
        self.assertEqual(result["request_id"], [0, 1])
        self.assertEqual(result["prompt"], ["first", "second"])
        self.assertEqual(result["output_tokens"], [3, 1])
        self.assertEqual(result["input_tokens"], [3, 3])
        self.assertEqual(result["batch_duration_s"], [2.5, 2.5])
        self.assertEqual(result["batch_size"], [2, 2])
        self.assertEqual(len(set(result["batch_id"])), 1)
        self.assertNotIn("processing_time", result)

    def test_batches_have_distinct_identity(self):
        predictor, llm, module = self.make_predictor()
        llm.generate.return_value = [response("first")]
        first = self.invoke(predictor, ("first",))
        second = self.invoke(predictor, ("first",))
        self.assertNotEqual(first["batch_id"], second["batch_id"])

    def test_model_options_are_explicit_and_do_not_widen_ray_gpu_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            predictor, llm, module = self.make_predictor(model_path="Llama-405B", model_kwargs={"dtype": "bfloat16"})
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(predictor.ready()["cuda_visible_devices"], "")
        config = module.LLM.call_args.kwargs
        self.assertNotIn("quantization", config)
        self.assertNotIn("max_model_len", config)
        self.assertFalse(config["enable_prefix_caching"])
        self.assertEqual(config["dtype"], "bfloat16")
        self.assertEqual(config["distributed_executor_backend"], "ray")
        self.assertEqual(config["seed"], 42)
        self.assertEqual((config["tensor_parallel_size"], config["pipeline_parallel_size"]), (2, 2))

    def test_model_seed_agrees_with_sampling_and_rejects_conflicts_before_allocation(self):
        predictor, llm, module = self.make_predictor(model_kwargs={"seed": 42})
        self.assertEqual(predictor.model_options["seed"], predictor.sampling_params.seed)
        module.LLM.reset_mock()
        with self.assertRaisesRegex(ValueError, "must match"):
            VLLMPredictor(model_path="model", tensor_parallel_size=1, pipeline_parallel_size=1,
                          sampling_params={"n": 1, "seed": 42}, model_kwargs={"seed": 43})
        module.LLM.assert_not_called()

    def test_warmup_does_not_mutate_benchmark_sampling(self):
        predictor, llm, module = self.make_predictor()
        llm.generate.return_value = [response("Hello", (20,))]
        original = predictor.sampling_params
        with patch("tokenpowerbench.distributed.predictor.time.perf_counter", side_effect=[1.0, 2.0]):
            report = predictor.warmup()
        params = llm.generate.call_args.args[1]
        self.assertIsNot(params, original)
        self.assertEqual((params.max_tokens, params.temperature, params.seed), (1, 0, 42))
        self.assertEqual(original.max_tokens, 32)
        self.assertEqual(report["output_tokens"], [1])

    def test_invalid_or_incomplete_outputs_fail_instead_of_counting_success(self):
        predictor, llm, module = self.make_predictor()
        invalid = [[], [response("first")], [response("first", finished=False), response("second")],
                   [response("wrong"), response("second")],
                   [response("first", ()), response("second")]]
        missing_ids = response("first")
        del missing_ids.outputs[0].token_ids
        invalid.append([missing_ids, response("second")])
        two_completions = response("first")
        two_completions.outputs.append(two_completions.outputs[0])
        invalid.append([two_completions, response("second")])
        aborted = response("first")
        aborted.outputs[0].finish_reason = "abort"
        invalid.append([aborted, response("second")])
        for outputs in invalid:
            with self.subTest(outputs=outputs), self.assertRaises(RuntimeError):
                llm.generate.return_value = outputs
                self.invoke(predictor)

    def test_generation_error_propagates(self):
        predictor, llm, module = self.make_predictor()
        error = RuntimeError("GPU worker lost")
        llm.generate.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.invoke(predictor)
        self.assertIs(caught.exception, error)

    def test_invalid_inputs_fail_before_generation(self):
        predictor, llm, module = self.make_predictor()
        cases = [{"text": ["a"]}, {"text": "a", "request_id": [1]},
                 {"text": [], "request_id": []}, {"text": [""], "request_id": [1]},
                 {"text": ["a", "b"], "request_id": [1]},
                 {"text": ["a", "b"], "request_id": [1, 1]},
                 {"text": ["a"], "request_id": [True]}]
        for batch in cases:
            with self.subTest(batch=batch), self.assertRaises((ValueError, TypeError)):
                predictor(batch)
        llm.generate.assert_not_called()

    def test_invalid_constructor_options_fail_before_model_allocation(self):
        module = SimpleNamespace(LLM=Mock(), SamplingParams=SamplingParams)
        with patch.dict(sys.modules, {"vllm": module}):
            for changes in ({"tensor_parallel_size": 0}, {"pipeline_parallel_size": True},
                            {"sampling_params": {"n": 2}}, {"model_kwargs": {"model": "other"}}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    VLLMPredictor(**{"model_path": "test", "tensor_parallel_size": 1,
                                     "pipeline_parallel_size": 1, "sampling_params": {"n": 1}, **changes})
        module.LLM.assert_not_called()

    def test_close_releases_v1_and_rejects_further_use(self):
        predictor, llm, module = self.make_predictor()
        shutdown = llm.llm_engine.engine_core.shutdown
        predictor.close()
        predictor.close()
        shutdown.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            predictor.ready()

    def test_close_releases_v0_executor(self):
        predictor, llm, module = self.make_predictor()
        shutdown = Mock()
        llm.llm_engine = SimpleNamespace(model_executor=SimpleNamespace(shutdown=shutdown))
        predictor.close()
        shutdown.assert_called_once_with()
        self.assertIsNone(llm.llm_engine.model_executor)

    def test_import_never_imports_inference_or_cluster_dependencies(self):
        code = "from tokenpowerbench.distributed import RayClusterConfig, VLLMPredictor; import sys; assert not {'torch','vllm','ray','numpy'}.intersection(sys.modules)"
        result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

"""
VLLMPredictor — Ray batch-processing worker for distributed vLLM inference.

Each worker process owns one vLLM LLM instance and processes batches of
prompts assigned to it by Ray Data's map_batches API.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
from vllm import LLM, SamplingParams


# Model-size detection keywords → vLLM constructor kwargs
_MODEL_CONFIGS: List[tuple] = [
    # (keyword_in_model_name, kwargs)
    ("405b", dict(
        quantization="fp8",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
    )),
    ("70b", dict(
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
    )),
    # DeepSeek MoE — fp8 recommended
    ("deepseek", dict(
        quantization="fp8",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
    )),
    # Default for smaller models (≤ 8B)
    ("", dict(
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        max_num_batched_tokens=4096,
        max_num_seqs=16,
    )),
]

_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


class VLLMPredictor:
    """
    Ray actor class: initialises a vLLM model and processes prompt batches.

    Instantiated once per Ray worker by map_batches; __call__ is invoked
    for every batch assigned to that worker.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        sampling_params: SamplingParams,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.sampling_params = sampling_params
        self.verbose = verbose
        self._batches_processed = 0
        self._first_token_recorded = False

        self._init_model()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        # Remove any leftover CUDA device restrictions from the parent env
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        model_key = os.path.basename(self.model_path).lower()
        extra_kwargs = self._model_kwargs(model_key)

        print(f"[VLLMPredictor] Loading model: {self.model_path}")
        print(f"  TP={self.tensor_parallel_size}  PP={self.pipeline_parallel_size}")
        print(f"  Extra kwargs: {extra_kwargs}")

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            trust_remote_code=True,
            enforce_eager=True,
            enable_prefix_caching=True,
            distributed_executor_backend="ray",
            load_format="auto",
            dtype="auto",
            **extra_kwargs,
        )
        print("[VLLMPredictor] Model loaded successfully.")

    @staticmethod
    def _model_kwargs(model_key: str) -> dict:
        for keyword, kwargs in _MODEL_CONFIGS:
            if keyword and keyword in model_key:
                return dict(kwargs)  # copy
        return dict(_MODEL_CONFIGS[-1][1])  # default / small model

    # ------------------------------------------------------------------
    # Batch processing (called by Ray Data)
    # ------------------------------------------------------------------

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        prompts_in_batch = batch["text"]
        self._batches_processed += 1

        prompts_out: List[str] = []
        generated_texts: List[str] = []
        processing_times: List[float] = []

        for i, raw_instruction in enumerate(prompts_in_batch):
            global_idx = (self._batches_processed - 1) * len(prompts_in_batch) + i + 1
            formatted = _PROMPT_TEMPLATE.format(instruction=raw_instruction)

            t0 = time.time()
            outputs = self.llm.generate([formatted], self.sampling_params)
            elapsed = time.time() - t0

            response = ""
            if outputs and outputs[0].outputs:
                response = " ".join(o.text for o in outputs[0].outputs)
                if not self._first_token_recorded and response.strip():
                    self._first_token_recorded = True

            prompts_out.append(raw_instruction)
            generated_texts.append(response)
            processing_times.append(elapsed)

            if self.verbose:
                tps = len(response.split()) / elapsed if elapsed > 0 else 0
                print(
                    f"[VLLMPredictor] prompt #{global_idx}: "
                    f"{elapsed:.2f}s  {tps:.1f} tok/s"
                )

        return {
            "prompt": prompts_out,
            "generated_text": generated_texts,
            "processing_time": processing_times,
        }

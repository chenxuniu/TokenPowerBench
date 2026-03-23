"""
Dataset loader supporting Alpaca, Dolly 15K, LongBench, and HumanEval.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

try:
    from datasets import load_dataset as hf_load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

_SUPPORTED = ("alpaca", "dolly", "longbench", "humaneval")


class DatasetLoader:
    """
    Load and pre-process prompts from standard LLM evaluation datasets.

    Parameters
    ----------
    cache_dir : str, optional
        HuggingFace dataset cache directory.
    seed : int
        Random seed for reproducible sampling.
    """

    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42) -> None:
        self.cache_dir = cache_dir
        self.seed = seed
        random.seed(seed)
        if not _HF_AVAILABLE:
            print(
                "Warning: HuggingFace `datasets` library not installed. "
                "Falling back to built-in prompts. Run: pip install datasets"
            )

    def load(
        self,
        dataset: str,
        num_samples: int = 1000,
        min_words: int = 5,
        max_words: int = 100,
    ) -> List[str]:
        """
        Load, filter, and sample prompts.

        Parameters
        ----------
        dataset : str
            One of: "alpaca", "dolly", "longbench", "humaneval".
        num_samples : int
            Maximum number of prompts to return.
        min_words, max_words : int
            Filter by prompt length (word count).

        Returns
        -------
        List[str]
            Sampled prompts ready for inference.
        """
        name = dataset.lower().strip()
        loaders = {
            "alpaca": self._alpaca,
            "dolly": self._dolly,
            "longbench": self._longbench,
            "humaneval": self._humaneval,
        }
        if name not in loaders:
            raise ValueError(
                f"Unknown dataset {name!r}. Supported: {_SUPPORTED}"
            )
        return loaders[name](num_samples, min_words, max_words)

    @staticmethod
    def supported_datasets() -> Dict[str, str]:
        return {
            "alpaca": "Stanford Alpaca — 52K instruction-following demos",
            "dolly": "Databricks Dolly 15K — high-quality instruction data",
            "longbench": "LongBench — long-context multi-task benchmark",
            "humaneval": "HumanEval — Python code completion tasks",
        }

    # ------------------------------------------------------------------
    # Dataset-specific loaders
    # ------------------------------------------------------------------

    def _alpaca(self, n: int, min_w: int, max_w: int) -> List[str]:
        if not _HF_AVAILABLE:
            return self._fallback()
        try:
            ds = hf_load_dataset("tatsu-lab/alpaca", cache_dir=self.cache_dir)
            prompts = []
            for item in ds.get("train", []):
                instr = item.get("instruction", "").strip()
                ctx = item.get("input", "").strip()
                if not instr:
                    continue
                prompts.append(f"{instr}\n\nContext: {ctx}" if ctx else instr)
            return self._filter_sample(prompts, n, min_w, max_w, "Alpaca")
        except Exception as exc:
            print(f"[DatasetLoader] Alpaca load failed: {exc}")
            return self._fallback()

    def _dolly(self, n: int, min_w: int, max_w: int) -> List[str]:
        if not _HF_AVAILABLE:
            return self._fallback()
        try:
            ds = hf_load_dataset(
                "databricks/databricks-dolly-15k", cache_dir=self.cache_dir
            )
            prompts = []
            for item in ds.get("train", []):
                instr = item.get("instruction", "").strip()
                ctx = item.get("context", "").strip()
                if not instr:
                    continue
                prompts.append(f"{instr}\n\nContext: {ctx}" if ctx else instr)
            return self._filter_sample(prompts, n, min_w, max_w, "Dolly 15K")
        except Exception as exc:
            print(f"[DatasetLoader] Dolly load failed: {exc}")
            return self._fallback()

    def _longbench(self, n: int, min_w: int, max_w: int) -> List[str]:
        if not _HF_AVAILABLE:
            return self._longbench_fallback()
        subtasks = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]
        prompts = []
        for sub in subtasks:
            try:
                ds = hf_load_dataset("THUDM/LongBench", sub, cache_dir=self.cache_dir)
                for item in ds.get("test", []):
                    text = item.get("input", "").strip()
                    if text:
                        prompts.append(text)
            except Exception as exc:
                print(f"[DatasetLoader] LongBench/{sub} failed: {exc}")
        if not prompts:
            return self._longbench_fallback()
        return self._filter_sample(prompts, n, min_w, max_w, "LongBench")

    def _humaneval(self, n: int, min_w: int, max_w: int) -> List[str]:
        if not _HF_AVAILABLE:
            return self._humaneval_fallback()
        try:
            ds = hf_load_dataset(
                "openai/openai_humaneval", cache_dir=self.cache_dir
            )
            prompts = [
                f"Complete the following Python function:\n\n{item['prompt']}"
                for item in ds.get("test", [])
                if item.get("prompt", "").strip()
            ]
            return self._filter_sample(prompts, n, min_w, max_w, "HumanEval")
        except Exception as exc:
            print(f"[DatasetLoader] HumanEval load failed: {exc}")
            return self._humaneval_fallback()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_sample(
        self, prompts: List[str], n: int, min_w: int, max_w: int, name: str
    ) -> List[str]:
        filtered = [p for p in prompts if min_w <= len(p.split()) <= max_w]
        print(
            f"[DatasetLoader] {name}: {len(prompts)} raw → "
            f"{len(filtered)} after length filter ({min_w}–{max_w} words)"
        )
        if n < len(filtered):
            sampled = random.sample(filtered, n)
            print(f"[DatasetLoader] Sampled {n} from {len(filtered)}")
            return sampled
        print(f"[DatasetLoader] Using all {len(filtered)} prompts")
        return filtered

    @staticmethod
    def _fallback() -> List[str]:
        return [
            "Explain the concept of machine learning.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does artificial intelligence work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain quantum computing in simple terms.",
            "What are the main causes of climate change?",
            "How do neural networks learn?",
        ]

    @staticmethod
    def _longbench_fallback() -> List[str]:
        return [
            (
                "Given the following passage about climate change, analyze the main "
                "arguments and provide a comprehensive summary:\n\n"
                "Climate change refers to long-term changes in global and regional "
                "climate patterns. The primary driver of modern climate change is "
                "human activity, particularly the emission of greenhouse gases such "
                "as carbon dioxide and methane."
            ),
        ]

    @staticmethod
    def _humaneval_fallback() -> List[str]:
        return [
            "Complete the following Python function:\n\n"
            "def fibonacci(n: int) -> int:\n"
            '    """Return the nth Fibonacci number."""\n'
            "    # Your code here",
        ]

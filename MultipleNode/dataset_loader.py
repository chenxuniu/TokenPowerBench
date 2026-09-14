"""Source-checkout dataset API backed by the shared strict dataset loader."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DatasetLoader:
    """Load actual dataset prompts with reproducible, instance-local sampling.

    Dataset dependencies are loaded on demand. Missing datasets and download
    failures propagate; synthetic prompts are never substituted.
    """

    def __init__(self, cache_dir=None, seed=42):
        self.cache_dir = cache_dir
        self.seed = seed
        self._loader = None

    def _shared_loader(self):
        if self._loader is None:
            from tokenpowerbench.data import DatasetLoader as SharedLoader
            self._loader = SharedLoader(cache_dir=self.cache_dir, seed=self.seed)
        return self._loader

    def load_dataset(self, dataset_name, num_samples=1000, min_length=5, max_length=100):
        return self._shared_loader().load(dataset_name, num_samples=num_samples,
                                          min_words=min_length, max_words=max_length)

    def load(self, dataset, num_samples=1000, min_words=5, max_words=100):
        return self._shared_loader().load(dataset, num_samples=num_samples,
                                          min_words=min_words, max_words=max_words)

    def get_dataset_info(self):
        return self._shared_loader().supported_datasets()

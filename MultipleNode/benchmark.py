#!/usr/bin/env python3
"""Run the shared multi-node CLI from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_multi_node import run


if __name__ == "__main__":
    if any(argument == "--timestamps-dir" or argument.startswith("--timestamps-dir=") for argument in sys.argv[1:]):
        raise SystemExit("Use --output-dir for benchmark artifacts; --timestamps-dir is unsupported.")
    raise SystemExit(run())

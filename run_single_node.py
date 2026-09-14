#!/usr/bin/env python3
"""Run the TokenPowerBench single-node command-line interface."""

from tokenpowerbench.cli import main

run = main

if __name__ == "__main__":
    raise SystemExit(main())

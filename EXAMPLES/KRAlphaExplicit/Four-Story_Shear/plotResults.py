#!/usr/bin/env python3
"""Generate CUDA comparison figures for the four-story shear frame."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotResults import _parse_plot_jobs, run

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    jobs = _parse_plot_jobs(sys.argv[1:])
    raise SystemExit(run(here, jobs=jobs))

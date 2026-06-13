#!/usr/bin/env python3
"""Generate woodbury-style figures for the Two-Story MRF KRAlphaExplicit sweep."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotResults import _parse_plot_jobs, run

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    jobs = _parse_plot_jobs(sys.argv[1:])
    sys.exit(run(here, jobs=jobs))

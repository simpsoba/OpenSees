# -*- coding: utf-8 -*-
"""
Forty-story shear frame — same integrator matrix as Two-Story_MRF (ops-cuda).

Usage:
  python3 run_integrators.py
  python3 run_integrators.py 0.75 1.0
  python3 run_integrators.py --jobs auto 0.0 0.25 0.5 0.75 0.9 0.99 0.999 1.0
  python3 run_integrators.py --plots-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shear_run_matrix import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)

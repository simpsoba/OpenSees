# -*- coding: utf-8 -*-
"""Shared helpers for KRAlphaExplicit example drivers."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List

import numpy as np


def permuted_node_tags(n_nodes: int, seed: int) -> List[int]:
    """Return a random permutation of tags 0..n_nodes-1 (fixed seed for reproducibility)."""
    tags = np.arange(n_nodes, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(tags)
    return tags.tolist()


def write_timing(output_folder: str, elapsed_s: float, *, label: str = "transient") -> None:
    """Write wall-clock seconds for an analysis leg to timing.txt."""
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, "timing.txt")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {stamp}\n")
        fh.write(f"label {label}\n")
        fh.write(f"wall_time_s {elapsed_s:.6f}\n")
    results_path = os.path.join(output_folder, "results.txt")
    with open(results_path, "a+", encoding="utf-8") as fh:
        fh.write(f"{stamp} - {label} wall time: {elapsed_s:.6f} s\n")


@contextmanager
def timed_analysis(output_folder: str, *, label: str = "transient") -> Iterator[None]:
    """Measure and save wall time for a transient analysis block."""
    t0 = time.perf_counter()
    yield
    write_timing(output_folder, time.perf_counter() - t0, label=label)

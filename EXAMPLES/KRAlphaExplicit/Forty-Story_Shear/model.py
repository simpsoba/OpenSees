"""40-story shear frame with linearly varying stiffness (modalDampingW woodbury example)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis_utils import permuted_node_tags  # noqa: E402


def build(ops, nstories, mass, k_bottom, k_top, uy, b, *, tag_seed: int = 0):
    """
    Story Steel01 springs with k linear from base to roof.

    Story index i (0=base .. nstories=roof) uses a random permutation of 0..nstories.
    """
    n_nodes = nstories + 1
    s = permuted_node_tags(n_nodes, tag_seed)

    ops.node(s[0], 0)
    ops.fix(s[0], 1)
    for i in range(1, nstories + 1):
        ops.node(s[i], 0)
        ops.mass(s[i], mass)
        k = k_bottom + (k_top - k_bottom) * (i - 1) / (nstories - 1)
        fy = k * uy
        ops.uniaxialMaterial("Steel01", i, fy, k, b)
        ops.element("zeroLength", i, s[i - 1], s[i], "-mat", i, "-dir", 1)
    return list(s)

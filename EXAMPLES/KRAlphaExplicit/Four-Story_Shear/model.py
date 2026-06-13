"""Four-story shear frame (Scott 2019 / modalDampingW woodbury example)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis_utils import permuted_node_tags  # noqa: E402


def build(ops, k, m, uy, b, *, tag_seed: int = 0):
    """
    Steel01 zeroLength springs, Scott story connectivity.

    Story index i (0=base .. 4=roof) is tagged with a random permutation of 0..4.
    Returned list gives OpenSees node tags in story order for recorders.
    """
    n_nodes = 5
    s = permuted_node_tags(n_nodes, tag_seed)
    fy = k * uy

    ops.node(s[0], 0)
    ops.fix(s[0], 1)
    ops.node(s[1], 0)
    ops.mass(s[1], m)
    ops.node(s[2], 0)
    ops.mass(s[2], m)
    ops.node(s[3], 0)
    ops.mass(s[3], m)
    ops.node(s[4], 0)
    ops.mass(s[4], 0.5 * m)

    ops.uniaxialMaterial("Steel01", 1, fy, k, b)
    ops.element("zeroLength", 1, s[0], s[1], "-mat", 1, "-dir", 1)
    ops.element("zeroLength", 2, s[1], s[3], "-mat", 1, "-dir", 1)
    ops.element("zeroLength", 3, s[3], s[2], "-mat", 1, "-dir", 1)
    ops.element("zeroLength", 4, s[3], s[4], "-mat", 1, "-dir", 1)
    return list(s)

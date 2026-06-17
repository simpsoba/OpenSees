# SDOF free-vibration settings (m=1, Tn=2 s, zeta=0, F=0)

import math

MASS = 1.0
TN = 2.0
ZETA = 0.0
FORCE = 0.0

# Fixed physical horizon for every Δt sweep (10 s ≈ 5 periods at Tn=2 s).
T_FINAL = 10.0
DT_THEORY = 0.005
DT_PLOT = 0.05  # subsample marker/lines on plots

DT_CASES = (
    {"dt": 0.2, "tag": "dt_0.2"},
    {"dt": 0.10, "tag": "dt_0.10"},
    {"dt": 0.05, "tag": "dt_0.05"},
    {"dt": 0.01, "tag": "dt_0.01"},
)

# Reference Δt (Tn/10) — used for default single-dt runs / docs.
DT_ANALYSIS = DT_CASES[0]["dt"]

# u0=0, v0=ωn -> peak |u|≈1 for ζ=0 (same amplitude scale as init_disp); fixed for all Δt.
INIT_VEL_V0 = 2.0 * math.pi / TN

IC_CASES = (
    {"tag": "init_disp", "u0": 1.0, "v0": 0.0},
    {"tag": "init_vel", "u0": 0.0, "v0": INIT_VEL_V0},
)

DEFAULT_RHOS = (0.0, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 1.0)

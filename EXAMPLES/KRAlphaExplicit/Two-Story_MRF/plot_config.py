# Plot / analysis settings for Two-Story MRF (must match two_story_MRF.MONITOR_NODES)
DT_ANALYSIS = 0.005  # fixed integrator step (s); same for all integrators
FREE_VIBRATION_SECONDS = 5.0  # extra duration after ground motion (s)
DT_PLOT = DT_ANALYSIS  # history plot decimation (s); keep equal to DT_ANALYSIS

# Default ρ sweep for run_integrators.py (matches EXAMPLES/KRAlphaExplicit/plotResults.py RHOS)
DEFAULT_RHOS = [0.0, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 1.0]

RECORD_NODES = [17, 10, 2]
MONITOR_DOFS = [1, 2, 3]
HORIZONTAL_DOF = 1  # used for floor history / error / drift plots

PROFILE_FLOORS = [0, 1, 2]
PROFILE_YLABEL = "floor"

PLOT_FLOOR_NODES = (10, 2)
FLOOR_LABELS = ("floor 1", "roof (node 2)")

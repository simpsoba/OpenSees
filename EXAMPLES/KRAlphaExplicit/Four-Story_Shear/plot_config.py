# Plot / analysis settings (recorder columns follow story order, not OpenSees tag order)

NODE_TAG_SEED = 42
DT_ANALYSIS = 0.02
DT_PLOT = DT_ANALYSIS
GM_DT = 0.02
GM_SCALE = 9.81
FREE_VIBRATION_SECONDS = 0.0  # 0 → free tail = full excitation duration (50 s)

# Story indices 0..4 (column j in disp.out = story j)
RECORD_NODES = [0, 1, 2, 3, 4]
MONITOR_DOFS = [1]
HORIZONTAL_DOF = 1

PROFILE_FLOORS = [0, 1, 2, 3, 4]
PROFILE_YLABEL = "floor"

PLOT_FLOOR_NODES = (1, 3, 2, 4)
FLOOR_LABELS = ("floor 1", "floor 3", "floor 2", "floor 4 (roof)")

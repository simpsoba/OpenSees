# Plot / analysis settings for forty-story shear frame (recorder columns = story order)

NODE_TAG_SEED = 42
DT_ANALYSIS = 0.02
DT_PLOT = DT_ANALYSIS
GM_DT = 0.02
GM_SCALE = 9.81
FREE_VIBRATION_SECONDS = 0.0

RECORD_NODES = list(range(41))
MONITOR_DOFS = [1]
HORIZONTAL_DOF = 1

PROFILE_FLOORS = list(range(41))
PROFILE_YLABEL = "floor"

PLOT_FLOOR_NODES = (1, 10, 20, 40)
FLOOR_LABELS = ("floor 1", "floor 10", "floor 20", "floor 40 (roof)")

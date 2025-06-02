import os

# ─── Project Directories ─────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train_FD001.txt")

# ─── Feature Engineering ─────────────────────────────────────────────────────

ROLLING_WINDOW = 5
RUL_THRESHOLD = 15

# ─── Unsupervised Anomaly Detection ──────────────────────────────────────────

IF_CONTAMINATION = 0.02
LOF_CONTAMINATION = 0.02
OCSVM_CONTAMINATION = 0.02

IF_N_ESTIMATORS = [50, 100, 150]
IF_CONTAM_OPTIONS = [0.01, 0.02, 0.05]

# ─── Supervised Classification ────────────────────────────────────────────────

RF_N_ESTIMATORS = [50, 100, 200]
RF_MAX_DEPTH = [None, 10, 20]
RF_MIN_SAMPLES_SPLIT = [2, 5, 10]

# ─── Plotting ─────────────────────────────────────────────────────────────────

PLOT_DPI = 100
MODEL_UNIT_TO_PLOT = 5  # <— This was missing before

# ─── General ─────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.2

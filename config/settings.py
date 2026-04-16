# --- Model ---
MODEL_PATH = "models/violence_model.h5"
THRESHOLD = 0.5

# --- Image / Video ---
IMG_SIZE = (224, 224)
FRAME_SIZE = (224, 224)   # alias kept for backward compat
FRAME_SKIP = 5            # process every Nth frame

# --- Training ---
BATCH_SIZE = 16
NUM_CLASSES = 14

# --- Dataset ---
DATASET_PATH = "datasets/ucf_crime"
PROCESSED_PATH = "datasets/processed"

# --- Splits ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# --- Optical flow ---
FLOW_OUTPUT_PATH = "datasets/optical_flow"

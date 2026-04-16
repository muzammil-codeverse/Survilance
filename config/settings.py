import os

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
# On Google Colab (Drive mounted):
#   BASE_PATH = "/content/drive/MyDrive/FYP_Violence_Detection"
# Locally (override with env var FYP_BASE_PATH if needed):
BASE_PATH = os.environ.get(
    "FYP_BASE_PATH",
    "/content/drive/MyDrive/FYP_Violence_Detection",
)

DATASET_PATH = os.path.join(BASE_PATH, "datasets")
PROCESSED_PATH = os.path.join(BASE_PATH, "processed")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models")
FLOW_OUTPUT_PATH = os.path.join(PROCESSED_PATH, "flow")

# Local model reference (inference only)
MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "violence_model.h5")

# ---------------------------------------------------------------------------
# Image / video
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
FRAME_SIZE = IMG_SIZE          # alias kept for backward compat
FRAME_SKIP = 5                 # process every Nth frame

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 16           # frames per clip fed to ViT (Phase 3)
BATCH_SIZE = 16
NUM_CLASSES = 14
LEARNING_RATE = 1e-4
THRESHOLD = 0.5                # binary inference threshold

# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# ---------------------------------------------------------------------------
# Constants only — NO paths here.
# All paths are passed as CLI arguments to each script.
# ---------------------------------------------------------------------------

# Image / video
IMG_SIZE = (224, 224)
FRAME_SKIP = 5              # process every Nth frame

# Model / training
SEQUENCE_LENGTH = 16        # frames per clip fed to ViT (Phase 3)
BATCH_SIZE = 16
NUM_CLASSES = 14
LEARNING_RATE = 1e-4
THRESHOLD = 0.5             # binary inference threshold

# Dataset splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

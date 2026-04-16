"""
validate_dataset.py
-------------------
Validates dataset integrity before any training run.

Checks performed:
  1. Class folders exist and are non-empty (frames)
  2. Every optical-flow .npy file is loadable and has the correct shape
  3. train / val / test splits exist and each class is represented
  4. label_map.json is present and consistent with split contents

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed (errors printed to stdout).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import PROCESSED_PATH, FLOW_OUTPUT_PATH, IMG_SIZE
from utils.logger import get_logger, log_stage

logger = get_logger("validate_dataset")

FRAMES_DIR = os.path.join(PROCESSED_PATH, "frames")
LABEL_MAP_PATH = os.path.join(PROCESSED_PATH, "label_map.json")
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_label_map() -> list[str]:
    errors = []
    if not os.path.isfile(LABEL_MAP_PATH):
        errors.append(f"MISSING label_map.json at {LABEL_MAP_PATH}")
        return errors

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    if not isinstance(label_map, dict) or not label_map:
        errors.append("label_map.json is empty or malformed")
        return errors

    labels = list(label_map.values())
    if sorted(labels) != list(range(len(labels))):
        errors.append(f"label_map indices are not contiguous 0..N-1: {labels}")

    logger.info(f"[label_map] OK — {len(label_map)} classes")
    return errors


def check_frames_dir() -> list[str]:
    errors = []

    if not os.path.isdir(FRAMES_DIR):
        errors.append(f"MISSING frames directory: {FRAMES_DIR}")
        return errors

    class_dirs = [
        d for d in os.listdir(FRAMES_DIR)
        if os.path.isdir(os.path.join(FRAMES_DIR, d))
    ]

    if not class_dirs:
        errors.append(f"No class sub-directories found in {FRAMES_DIR}")
        return errors

    for cls in class_dirs:
        cls_dir = os.path.join(FRAMES_DIR, cls)
        images = [
            f for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            errors.append(f"EMPTY class folder (no images): {cls_dir}")
        else:
            logger.info(f"[frames] {cls}: {len(images)} images OK")

    return errors


def check_optical_flow() -> list[str]:
    errors = []

    if not os.path.isdir(FLOW_OUTPUT_PATH):
        # Not fatal — flow may not be generated yet
        logger.warning(f"Optical flow directory not found: {FLOW_OUTPUT_PATH}  (skipping)")
        return errors

    class_dirs = [
        d for d in os.listdir(FLOW_OUTPUT_PATH)
        if os.path.isdir(os.path.join(FLOW_OUTPUT_PATH, d))
    ]

    total_ok = 0
    for cls in class_dirs:
        cls_dir = os.path.join(FLOW_OUTPUT_PATH, cls)
        npy_files = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]

        for npy_file in npy_files:
            path = os.path.join(cls_dir, npy_file)
            try:
                flow = np.load(path)
                # Expected shape: (H, W, 2)
                if flow.ndim != 3 or flow.shape[2] != 2:
                    errors.append(
                        f"BAD flow shape {flow.shape} in {path} (expected H×W×2)"
                    )
                else:
                    total_ok += 1
            except Exception as e:
                errors.append(f"CORRUPT flow file {path}: {e}")

    if total_ok:
        logger.info(f"[optical_flow] {total_ok} .npy files validated OK")

    return errors


def check_splits() -> list[str]:
    errors = []

    if not os.path.isfile(LABEL_MAP_PATH):
        errors.append("Cannot check splits — label_map.json missing")
        return errors

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    for split in SPLITS:
        split_dir = os.path.join(PROCESSED_PATH, split)

        if not os.path.isdir(split_dir):
            errors.append(f"MISSING split directory: {split_dir}")
            continue

        split_classes = set(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )
        expected = set(label_map.keys())
        missing = expected - split_classes

        if missing:
            errors.append(
                f"[{split}] Missing class folders: {sorted(missing)}"
            )

        total_samples = 0
        for cls in split_classes:
            cls_dir = os.path.join(split_dir, cls)
            n = len([
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ])
            if n == 0:
                errors.append(f"[{split}/{cls}] EMPTY — no images found")
            total_samples += n

        logger.info(f"[{split}] {total_samples} images across {len(split_classes)} classes")

    return errors


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_validation() -> bool:
    log_stage(logger, "Dataset Validation")

    all_errors: list[str] = []
    all_errors.extend(check_label_map())
    all_errors.extend(check_frames_dir())
    all_errors.extend(check_optical_flow())
    all_errors.extend(check_splits())

    if all_errors:
        logger.error(f"Validation FAILED — {len(all_errors)} issue(s) found:")
        for err in all_errors:
            logger.error(f"  ✗ {err}")
        return False

    logger.info("Validation PASSED — dataset is clean and ready for training.")
    return True


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)

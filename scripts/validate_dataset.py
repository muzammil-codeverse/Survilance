"""
validate_dataset.py
-------------------
Validate dataset integrity before any training run.

Checks:
  1. label_map.json exists and has contiguous 0..N-1 indices
  2. Frames directory exists with non-empty class folders
  3. Optical flow .npy files are loadable and have shape (H, W, 2)
  4. train / val / test splits exist and cover all classes

Exit code 0 = all checks passed.
Exit code 1 = one or more issues found.

Usage:
    python scripts/validate_dataset.py --input <processed_root> [--flow <flow_root>]

Example:
    python scripts/validate_dataset.py \
        --input /content/dataset \
        --flow  /content/flow
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger, log_stage

logger = get_logger("validate_dataset")

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dataset integrity before training."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Processed dataset root (contains label_map.json + train/val/test/).",
    )
    parser.add_argument(
        "--flow",
        default=None,
        help="(Optional) Optical flow root to validate .npy files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Individual checks — each receives explicit paths, zero global state
# ---------------------------------------------------------------------------

def check_label_map(processed_root: str) -> list[str]:
    errors = []
    path = os.path.join(processed_root, "label_map.json")

    if not os.path.isfile(path):
        errors.append(f"MISSING label_map.json at {path}")
        return errors

    with open(path) as f:
        label_map = json.load(f)

    if not isinstance(label_map, dict) or not label_map:
        errors.append(f"label_map.json is empty or malformed: {path}")
        return errors

    labels = list(label_map.values())
    if sorted(labels) != list(range(len(labels))):
        errors.append(f"label_map indices not contiguous 0..N-1: {labels}")
    else:
        logger.info(f"[label_map] OK — {len(label_map)} classes")

    return errors


def check_frames_dir(processed_root: str) -> list[str]:
    """
    Validates the processed split directories (train/val/test) for non-empty
    class folders.  The raw frames dir is no longer required — UCF-Crime ships
    pre-extracted PNGs, so extract_frames.py is not part of the pipeline.
    """
    errors = []

    # Check each split for at least one class with images
    for split in SPLITS:
        split_dir = os.path.join(processed_root, split)
        if not os.path.isdir(split_dir):
            continue   # missing splits caught by check_splits()

        class_dirs = [
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]

        for cls in class_dirs:
            cls_dir = os.path.join(split_dir, cls)
            images = [
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ]
            if not images:
                errors.append(f"EMPTY class folder in {split}: {cls_dir}")

    if not errors:
        logger.info("[frames] All split class folders contain images — OK")

    return errors


def check_optical_flow(flow_root: str) -> list[str]:
    errors = []

    if not os.path.isdir(flow_root):
        logger.warning(f"[flow] directory not found (skipping): {flow_root}")
        return errors

    class_dirs = [
        d for d in os.listdir(flow_root)
        if os.path.isdir(os.path.join(flow_root, d))
    ]

    total_ok = 0
    for cls in class_dirs:
        cls_dir = os.path.join(flow_root, cls)
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(cls_dir, fname)
            try:
                flow = np.load(path)
                if flow.ndim != 3 or flow.shape[2] != 2:
                    errors.append(f"BAD flow shape {flow.shape}: {path}")
                else:
                    total_ok += 1
            except Exception as e:
                errors.append(f"CORRUPT flow file {path}: {e}")

    if total_ok:
        logger.info(f"[flow] {total_ok} .npy files OK")

    return errors


def check_splits(processed_root: str) -> list[str]:
    errors = []
    label_map_path = os.path.join(processed_root, "label_map.json")

    if not os.path.isfile(label_map_path):
        errors.append("Cannot check splits — label_map.json missing")
        return errors

    with open(label_map_path) as f:
        label_map = json.load(f)

    expected_classes = set(label_map.keys())

    for split in SPLITS:
        split_dir = os.path.join(processed_root, split)

        if not os.path.isdir(split_dir):
            errors.append(f"MISSING split directory: {split_dir}")
            continue

        found_classes = {
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        }
        missing = expected_classes - found_classes
        if missing:
            errors.append(f"[{split}] Missing class folders: {sorted(missing)}")

        total = 0
        for cls in found_classes:
            cls_dir = os.path.join(split_dir, cls)
            n = len([
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ])
            if n == 0:
                errors.append(f"[{split}/{cls}] EMPTY — no images")
            total += n

        logger.info(f"[{split}] {total} images across {len(found_classes)} classes")

    return errors


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_validation(input_root: str, flow_root: str | None) -> bool:
    log_stage(logger, "Dataset Validation", f"input={input_root}")

    errors: list[str] = []
    errors.extend(check_label_map(input_root))
    errors.extend(check_frames_dir(input_root))

    if flow_root:
        errors.extend(check_optical_flow(flow_root))

    errors.extend(check_splits(input_root))

    if errors:
        logger.error(f"Validation FAILED — {len(errors)} issue(s):")
        for e in errors:
            logger.error(f"  x {e}")
        return False

    logger.info("Validation PASSED — dataset is clean and ready for training.")
    return True


if __name__ == "__main__":
    args = get_args()
    ok = run_validation(input_root=args.input, flow_root=args.flow)
    sys.exit(0 if ok else 1)

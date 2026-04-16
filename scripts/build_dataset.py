"""
build_dataset.py
----------------
Reads extracted frames from datasets/processed/frames/, creates a class→label
mapping, performs a stratified 70/20/10 train/val/test split, and writes the
final structure to datasets/processed/{train,val,test}/.

Output layout:
    datasets/processed/
        label_map.json          # {"Abuse": 0, "Arrest": 1, ...}
        train/
            Abuse/  *.jpg
            Arrest/ *.jpg
            ...
        val/
            ...
        test/
            ...
"""

import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    PROCESSED_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
)
from utils.logger import get_logger, log_stage, log_dataset_stats

logger = get_logger("build_dataset")

FRAMES_DIR = os.path.join(PROCESSED_PATH, "frames")
LABEL_MAP_PATH = os.path.join(PROCESSED_PATH, "label_map.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(42)  # reproducible splits


def _collect_samples(frames_root: str) -> tuple[dict, list]:
    """
    Returns:
        label_map  — {class_name: int}
        samples    — [(abs_path, label_int), ...]
    """
    class_dirs = sorted(
        d for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d))
    )

    label_map = {cls: idx for idx, cls in enumerate(class_dirs)}
    samples = []

    for cls, label in label_map.items():
        cls_dir = os.path.join(frames_root, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                samples.append((os.path.join(cls_dir, fname), label, cls))

    return label_map, samples


def _stratified_split(
    samples: list,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list, list, list]:
    """
    Stratify by class, then split each class into train/val/test.
    """
    # Group by class label
    class_buckets: dict[int, list] = {}
    for item in samples:
        label = item[1]
        class_buckets.setdefault(label, []).append(item)

    train, val, test = [], [], []

    for label, items in class_buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(items[:n_train])
        val.extend(items[n_train: n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def _copy_split(split_samples: list, split_name: str, output_root: str) -> None:
    """Copy files into output_root/{split_name}/{class_name}/."""
    for src_path, _label, cls_name in split_samples:
        dst_dir = os.path.join(output_root, split_name, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_path, dst_dir)


def build_dataset(frames_root: str, output_root: str) -> None:
    log_stage(logger, "Dataset Build", f"src={frames_root}  dst={output_root}")

    if not os.path.isdir(frames_root):
        logger.error(f"Frames directory not found: {frames_root}")
        sys.exit(1)

    label_map, samples = _collect_samples(frames_root)

    if not samples:
        logger.error("No image samples found. Run extract_frames.py first.")
        sys.exit(1)

    logger.info(f"Total samples: {len(samples)} | Classes: {len(label_map)}")

    # Persist label mapping
    os.makedirs(output_root, exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Label map saved → {LABEL_MAP_PATH}")

    train, val, test = _stratified_split(samples, TRAIN_RATIO, VAL_RATIO)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        _copy_split(split_data, split_name, output_root)
        log_dataset_stats(logger, split_name, len(split_data), list(label_map.keys()))

    logger.info("Dataset build complete.")
    logger.info(f"  train={len(train)}  val={len(val)}  test={len(test)}")


if __name__ == "__main__":
    build_dataset(FRAMES_DIR, PROCESSED_PATH)

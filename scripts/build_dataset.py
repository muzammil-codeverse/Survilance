"""
build_dataset.py
----------------
Merge frames (and optionally flow maps), create a label mapping, perform a
stratified 70/20/10 train/val/test split, and write the final split structure.

Usage:
    python scripts/build_dataset.py \
        --frames_input <frames_root> \
        --output       <output_root> \
        [--flow_input  <flow_root>]

Example:
    python scripts/build_dataset.py \
        --frames_input /content/frames \
        --flow_input   /content/flow \
        --output       /content/dataset
"""

import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import TRAIN_RATIO, VAL_RATIO
from utils.logger import get_logger, log_stage, log_dataset_stats

logger = get_logger("build_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(42)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stratified train/val/test splits from extracted frames."
    )
    parser.add_argument(
        "--frames_input",
        required=True,
        help="Root directory of extracted frames (output of extract_frames.py).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root directory to write train/val/test splits and label_map.json.",
    )
    parser.add_argument(
        "--flow_input",
        default=None,
        help="(Optional) Root directory of flow maps. Logged but not copied into splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Fraction of samples for training (default: {TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Fraction of samples for validation (default: {VAL_RATIO}).",
    )
    return parser.parse_args()


def _collect_samples(frames_root: str) -> tuple[dict, list]:
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
    class_buckets: dict[int, list] = {}
    for item in samples:
        class_buckets.setdefault(item[1], []).append(item)

    train, val, test = [], [], []

    for items in class_buckets.values():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train.extend(items[:n_train])
        val.extend(items[n_train: n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def _copy_split(split_samples: list, split_name: str, output_root: str) -> None:
    for src_path, _label, cls_name in split_samples:
        dst_dir = os.path.join(output_root, split_name, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_path, dst_dir)


def build_dataset(
    frames_input: str,
    output: str,
    flow_input: str | None,
    train_ratio: float,
    val_ratio: float,
) -> None:
    log_stage(logger, "Dataset Build", f"frames={frames_input}  out={output}")

    if flow_input:
        if os.path.isdir(flow_input):
            logger.info(f"Flow input registered: {flow_input}")
        else:
            logger.warning(f"Flow input not found (skipping): {flow_input}")

    if not os.path.isdir(frames_input):
        logger.error(f"Frames directory not found: {frames_input}")
        sys.exit(1)

    label_map, samples = _collect_samples(frames_input)

    if not samples:
        logger.error("No images found. Run extract_frames.py first.")
        sys.exit(1)

    logger.info(f"Samples: {len(samples)} | Classes: {len(label_map)}")

    os.makedirs(output, exist_ok=True)
    label_map_path = os.path.join(output, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"label_map.json saved → {label_map_path}")

    train, val, test = _stratified_split(samples, train_ratio, val_ratio)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        _copy_split(split_data, split_name, output)
        log_dataset_stats(logger, split_name, len(split_data), list(label_map.keys()))

    logger.info(f"Done. train={len(train)}  val={len(val)}  test={len(test)}")


if __name__ == "__main__":
    args = get_args()
    build_dataset(
        frames_input=args.frames_input,
        output=args.output,
        flow_input=args.flow_input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

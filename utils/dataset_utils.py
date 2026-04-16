"""
dataset_utils.py
----------------
Central dataset utility functions shared across scripts and training code.

Provides:
  - load_label_map()         load + validate label_map.json
  - scan_split()             list (path, label) pairs for a split
  - get_class_distribution() count samples per class in a split
  - compute_class_weights()  inverse-frequency weights for imbalanced datasets
"""

import json
import os
from collections import Counter

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger

logger = get_logger("dataset_utils")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_label_map(path: str) -> dict:
    """
    Load {class_name: int} from a JSON file.

    Raises:
        FileNotFoundError if the file does not exist.
        ValueError if the mapping is empty or malformed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"label_map.json not found at '{path}'. Run build_dataset.py first."
        )

    with open(path) as f:
        label_map = json.load(f)

    if not isinstance(label_map, dict) or not label_map:
        raise ValueError(f"label_map at '{path}' is empty or not a dict.")

    logger.debug(f"Loaded label_map: {len(label_map)} classes from {path}")
    return label_map


def scan_split(split_dir: str, label_map: dict) -> list[tuple[str, int]]:
    """
    Recursively scan `split_dir` and return a list of (abs_path, label_int)
    tuples for every recognised image file.

    Args:
        split_dir:  Path to a split directory (e.g. 'datasets/processed/train').
        label_map:  {class_name: int} mapping.

    Returns:
        Sorted list of (file_path, label) pairs.
    """
    samples = []

    for cls_name, label_idx in label_map.items():
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            logger.warning(f"Class directory not found: {cls_dir}")
            continue

        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                samples.append((os.path.join(cls_dir, fname), label_idx))

    logger.debug(f"Scanned {len(samples)} samples from {split_dir}")
    return samples


def get_class_distribution(split_dir: str, label_map: dict) -> dict:
    """
    Count how many samples exist per class in a split.

    Returns:
        {class_name: count}
    """
    samples = scan_split(split_dir, label_map)
    inv_map = {v: k for k, v in label_map.items()}
    counts = Counter(label for _, label in samples)
    return {inv_map[lbl]: cnt for lbl, cnt in sorted(counts.items())}


def compute_class_weights(split_dir: str, label_map: dict) -> dict:
    """
    Compute inverse-frequency class weights for imbalanced training.

    Formula: weight_i = total_samples / (num_classes * count_i)

    Returns:
        {label_int: weight_float}  — ready to pass to Keras class_weight arg.
    """
    distribution = get_class_distribution(split_dir, label_map)
    total = sum(distribution.values())
    num_classes = len(distribution)

    weights = {}
    for cls_name, count in distribution.items():
        label_idx = label_map[cls_name]
        weights[label_idx] = total / (num_classes * count) if count > 0 else 1.0

    logger.info("Class weights computed:")
    inv_map = {v: k for k, v in label_map.items()}
    for idx, w in sorted(weights.items()):
        logger.info(f"  [{idx:>2}] {inv_map[idx]:<20} weight={w:.4f}")

    return weights


def get_label_name(label_idx: int, label_map: dict) -> str:
    """Reverse lookup: int → class name."""
    inv_map = {v: k for k, v in label_map.items()}
    return inv_map.get(label_idx, f"unknown_{label_idx}")

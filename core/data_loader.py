"""
data_loader.py
--------------
Builds efficient tf.data.Dataset pipelines for the processed frame splits.

Usage:
    from core.data_loader import build_dataset

    train_ds = build_dataset("datasets/processed/train", label_map, training=True)
    val_ds   = build_dataset("datasets/processed/val",   label_map, training=False)
"""

import json
import os

import tensorflow as tf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import IMG_SIZE, BATCH_SIZE, PROCESSED_PATH
from utils.logger import get_logger

logger = get_logger("data_loader")

AUTOTUNE = tf.data.AUTOTUNE
LABEL_MAP_PATH = os.path.join(PROCESSED_PATH, "label_map.json")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_label_map(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"label_map.json not found at {path}. Run build_dataset.py first."
        )
    with open(path) as f:
        return json.load(f)


def _collect_file_label_pairs(split_dir: str, label_map: dict) -> tuple[list, list]:
    """Walk split_dir and return parallel lists of file paths and integer labels."""
    paths, labels = [], []

    for cls_name, label_idx in label_map.items():
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            logger.warning(f"Class directory missing: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label_idx)

    return paths, labels


# ---------------------------------------------------------------------------
# tf.data decode & augment functions (operate on tensors — no Python I/O)
# ---------------------------------------------------------------------------

def _decode_image(path: tf.Tensor, size: tuple = IMG_SIZE) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(
    split_dir: str,
    label_map: dict | None = None,
    training: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """
    Build a batched, prefetched tf.data.Dataset for a single split.

    Args:
        split_dir:      Path to the split directory (e.g. 'datasets/processed/train').
        label_map:      {class_name: int}. Loaded from disk if None.
        training:       If True, apply shuffling and augmentation.
        batch_size:     Samples per batch.
        shuffle_buffer: Shuffle buffer size (used only when training=True).

    Returns:
        A tf.data.Dataset yielding (image_batch, label_batch) tensors.
    """
    if label_map is None:
        label_map = _load_label_map(LABEL_MAP_PATH)

    paths, labels = _collect_file_label_pairs(split_dir, label_map)

    if not paths:
        raise ValueError(f"No samples found in {split_dir}. Check your dataset.")

    num_classes = len(label_map)
    logger.info(
        f"Building dataset | split={os.path.basename(split_dir)} "
        f"samples={len(paths)} classes={num_classes} training={training}"
    )

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Decode images in parallel
    image_ds = path_ds.map(
        lambda p: _decode_image(p, IMG_SIZE),
        num_parallel_calls=AUTOTUNE,
    )

    ds = tf.data.Dataset.zip((image_ds, label_ds))

    if training:
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda img, lbl: (_augment(img), lbl),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def load_all_splits(
    processed_root: str = PROCESSED_PATH,
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Convenience loader that returns (train_ds, val_ds, test_ds, label_map).
    """
    label_map = _load_label_map(LABEL_MAP_PATH)

    train_ds = build_dataset(
        os.path.join(processed_root, "train"),
        label_map=label_map,
        training=True,
        batch_size=batch_size,
    )
    val_ds = build_dataset(
        os.path.join(processed_root, "val"),
        label_map=label_map,
        training=False,
        batch_size=batch_size,
    )
    test_ds = build_dataset(
        os.path.join(processed_root, "test"),
        label_map=label_map,
        training=False,
        batch_size=batch_size,
    )

    return train_ds, val_ds, test_ds, label_map

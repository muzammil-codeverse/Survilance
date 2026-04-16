"""
build_dataset.py
----------------
Build stratified train/val/test splits from a UCF-Crime-style PNG frame dataset.

Expected input layout:
    INPUT/
        Abuse/
            Abuse001/
                frame001.png
                frame002.png
            Abuse002/
                ...
        Arrest/
            ...

Output layout:
    OUTPUT/
        label_map.json
        train/
            Abuse/   *.png   (frames copied flat per class)
            Arrest/
            ...
        val/
            ...
        test/
            ...

Split is performed at the VIDEO-FOLDER level (not frame level) to prevent
data leakage between splits.

Usage:
    python scripts/build_dataset.py \
        --input  <frames_root> \
        --output <output_root> \
        [--flow_input <flow_root>]

Example:
    python scripts/build_dataset.py \
        --input  /path/to/UCF_Crime_Frames \
        --output /path/to/dataset
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

random.seed(42)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stratified train/val/test splits from PNG frame sequences."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root directory with CLASS/VIDEO_FOLDER/frame.png structure.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root directory to write label_map.json + train/val/test splits.",
    )
    parser.add_argument(
        "--flow_input",
        default=None,
        help="(Optional) Flow root — logged for provenance, not copied into splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Fraction of videos for training (default: {TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Fraction of videos for validation (default: {VAL_RATIO}).",
    )
    return parser.parse_args()


def load_frames(folder: str) -> list[str]:
    """Return sorted list of image file paths inside `folder`."""
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def _collect_videos(input_root: str) -> tuple[dict, dict]:
    """
    Walk CLASS/VIDEO_FOLDER structure.

    Returns:
        label_map  — {class_name: int}
        videos     — {class_name: [video_folder_path, ...]}
    """
    class_dirs = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    label_map = {cls: idx for idx, cls in enumerate(class_dirs)}
    videos: dict[str, list] = {}

    for cls in class_dirs:
        cls_dir = os.path.join(input_root, cls)
        video_folders = sorted(
            os.path.join(cls_dir, d)
            for d in os.listdir(cls_dir)
            if os.path.isdir(os.path.join(cls_dir, d))
        )

        # Also handle flat layout (frames directly inside class folder)
        flat_frames = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]

        if video_folders:
            videos[cls] = video_folders
        elif flat_frames:
            # Treat the class folder itself as one "virtual" video
            videos[cls] = [cls_dir]
        else:
            logger.warning(f"No video folders or images found in: {cls_dir}")
            videos[cls] = []

    return label_map, videos


def _stratified_split(
    videos: dict,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list, list, list]:
    """
    Split at the video-folder level (prevents frame-level data leakage).

    Returns three lists of (video_folder_path, label_int, class_name).
    """
    train, val, test = [], [], []

    label_map_local = {cls: idx for idx, cls in enumerate(sorted(videos.keys()))}

    for cls, video_list in videos.items():
        label = label_map_local[cls]
        items = [(vf, label, cls) for vf in video_list]
        random.shuffle(items)

        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(items[:n_train])
        val.extend(items[n_train: n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def _copy_video_frames(video_path: str, cls_name: str, split_name: str, output_root: str) -> int:
    """
    Copy every frame from `video_path` into output_root/split_name/cls_name/.
    Prefix filenames with the video folder name to avoid collisions.

    Returns number of frames copied.
    """
    dst_dir = os.path.join(output_root, split_name, cls_name)
    os.makedirs(dst_dir, exist_ok=True)

    frames = load_frames(video_path)
    video_stem = os.path.basename(video_path)

    copied = 0
    for src in frames:
        fname = f"{video_stem}__{os.path.basename(src)}"
        shutil.copy2(src, os.path.join(dst_dir, fname))
        copied += 1

    return copied


def build_dataset(
    input_root: str,
    output: str,
    flow_input: str | None,
    train_ratio: float,
    val_ratio: float,
) -> None:
    log_stage(logger, "Dataset Build", f"input={input_root}  out={output}")

    if flow_input:
        logger.info(
            f"Flow input: {flow_input}"
            if os.path.isdir(flow_input)
            else f"Flow input not found (skipping): {flow_input}"
        )

    if not os.path.isdir(input_root):
        logger.error(f"Input directory not found: {input_root}")
        sys.exit(1)

    label_map, videos = _collect_videos(input_root)

    total_videos = sum(len(v) for v in videos.values())
    if total_videos == 0:
        logger.error("No video folders found. Check your --input path.")
        sys.exit(1)

    logger.info(f"Classes={len(label_map)} | Video folders={total_videos}")

    os.makedirs(output, exist_ok=True)
    label_map_path = os.path.join(output, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"label_map.json saved → {label_map_path}")

    train, val, test = _stratified_split(videos, train_ratio, val_ratio)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        total_frames = 0
        for video_path, _label, cls_name in split_data:
            total_frames += _copy_video_frames(video_path, cls_name, split_name, output)
        log_dataset_stats(logger, split_name, total_frames, list(label_map.keys()))

    logger.info(
        f"Done. Videos split — train={len(train)}  val={len(val)}  test={len(test)}"
    )


if __name__ == "__main__":
    args = get_args()
    build_dataset(
        input_root=args.input,
        output=args.output,
        flow_input=args.flow_input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

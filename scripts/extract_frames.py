"""
extract_frames.py
-----------------
DEPRECATED — not used in the current pipeline.

The UCF-Crime dataset is distributed as pre-extracted PNG frame sequences
(CLASS/VIDEO_FOLDER/frame.png). No video decoding is needed.

Current pipeline:
    optical_flow.py  →  build_dataset.py  →  validate_dataset.py

This script is kept only as a reference for datasets that ship raw .mp4/.avi
files. Do NOT call it as part of the standard pipeline.
-----------------
Original purpose: Extract frames from raw videos into a per-class output folder.

Usage:
    python scripts/extract_frames.py --input <raw_video_root> --output <frames_root>

Example:
    python scripts/extract_frames.py \
        --input  /content/data/ucf_crime \
        --output /content/frames
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import IMG_SIZE, FRAME_SKIP
from utils.logger import get_logger, log_stage

logger = get_logger("extract_frames")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from raw videos into per-class JPEG folders."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root directory containing per-class sub-folders of raw videos.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root directory to write extracted frames into.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=FRAME_SKIP,
        help=f"Save every Nth frame (default: {FRAME_SKIP}).",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=list(IMG_SIZE),
        metavar=("W", "H"),
        help=f"Output frame size width height (default: {IMG_SIZE[0]} {IMG_SIZE[1]}).",
    )
    return parser.parse_args()


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_skip: int,
    size: tuple,
) -> int:
    """
    Stream through a single video file and save every `frame_skip`-th frame.

    Returns:
        Number of frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return 0

    frame_idx = 0
    saved = 0
    stem = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame = cv2.resize(frame, tuple(size))
            filename = f"{stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def process_dataset(input_root: str, output_root: str, frame_skip: int, size: tuple) -> None:
    """
    Walk `input_root`, detect class folders, extract frames for every video.
    """
    log_stage(logger, "Frame Extraction", f"src={input_root}  dst={output_root}")

    if not os.path.isdir(input_root):
        logger.error(f"Input path not found: {input_root}")
        sys.exit(1)

    class_dirs = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    if not class_dirs:
        logger.error(f"No class sub-directories found in: {input_root}")
        sys.exit(1)

    logger.info(f"Found {len(class_dirs)} classes: {class_dirs}")

    total_frames = 0
    total_videos = 0

    for cls in class_dirs:
        cls_src = os.path.join(input_root, cls)
        cls_dst = os.path.join(output_root, cls)

        for dirpath, _, filenames in os.walk(cls_src):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in VIDEO_EXTENSIONS:
                    continue

                video_path = os.path.join(dirpath, fname)
                saved = extract_frames_from_video(video_path, cls_dst, frame_skip, size)
                total_frames += saved
                total_videos += 1
                logger.info(f"  [{cls}] {fname} → {saved} frames saved")

    logger.info(f"Done. Videos={total_videos} | Frames saved={total_frames}")


if __name__ == "__main__":
    args = get_args()
    process_dataset(
        input_root=args.input,
        output_root=args.output,
        frame_skip=args.frame_skip,
        size=tuple(args.size),
    )

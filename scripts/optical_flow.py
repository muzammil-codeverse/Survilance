"""
optical_flow.py
---------------
Compute dense Farneback optical flow from pre-extracted PNG/JPG frame sequences.

Expected input layout (UCF-Crime style — frames already on disk):
    INPUT/
        Abuse/
            Abuse001/
                frame001.png
                frame002.png
                ...
            Abuse002/
                ...
        Arrest/
            ...

Output layout (mirrors input structure):
    OUTPUT/
        Abuse/
            Abuse001/
                frame001_flow.npy   # flow from frame001 → frame002, shape (H,W,2)
                frame002_flow.npy
                ...

Usage:
    python scripts/optical_flow.py --input <frames_root> --output <flow_root>

Example:
    python scripts/optical_flow.py \
        --input  /content/UCF_Crime_Frames \
        --output /content/flow
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger, log_stage

logger = get_logger("optical_flow")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Farneback parameters — tuned for 224×224 frames
FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Farneback optical flow from PNG/JPG frame sequences."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root directory containing CLASS/VIDEO_FOLDER/frame.png structure.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root directory to write flow .npy maps (mirrors input structure).",
    )
    return parser.parse_args()


def load_frames(folder: str) -> list:
    """
    Load all PNG/JPG images from `folder` in sorted filename order.

    Returns:
        List of BGR numpy arrays. Empty list if no images found.
    """
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    frames = []
    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            logger.warning(f"Could not read image: {os.path.join(folder, fname)}")
            continue
        frames.append((fname, img))

    return frames   # [(filename, bgr_array), ...]


def compute_optical_flow(frames: list) -> list:
    """
    Compute Farneback dense optical flow between every consecutive frame pair.

    Args:
        frames: List of (filename, bgr_array) tuples, in temporal order.

    Returns:
        List of (src_stem, flow_array) where flow_array.shape == (H, W, 2).
    """
    results = []

    if len(frames) < 2:
        return results

    prev_gray = cv2.cvtColor(frames[0][1], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i][1], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **FB_PARAMS
        )  # (H, W, 2) float32

        src_stem = os.path.splitext(frames[i - 1][0])[0]
        results.append((src_stem, flow))

        prev_gray = curr_gray

    return results


def process_video_folder(video_folder: str, out_folder: str) -> int:
    """
    Process a single video folder: load frames → compute flow → save .npy files.

    Returns:
        Number of flow maps saved.
    """
    frames = load_frames(video_folder)

    if not frames:
        logger.warning(f"No images found in: {video_folder}")
        return 0

    if len(frames) < 2:
        logger.warning(f"Need ≥2 frames for flow, skipping: {video_folder}")
        return 0

    flow_pairs = compute_optical_flow(frames)
    os.makedirs(out_folder, exist_ok=True)

    for src_stem, flow in flow_pairs:
        np.save(os.path.join(out_folder, f"{src_stem}_flow.npy"), flow)

    return len(flow_pairs)


def process_optical_flow(input_root: str, output_root: str) -> None:
    """
    Walk CLASS/VIDEO_FOLDER structure under `input_root` and compute flow for
    every video folder.
    """
    log_stage(logger, "Optical Flow Computation", f"src={input_root}  dst={output_root}")

    if not os.path.isdir(input_root):
        logger.error(f"Input directory not found: {input_root}")
        sys.exit(1)

    class_dirs = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    if not class_dirs:
        logger.error(f"No class sub-directories in: {input_root}")
        sys.exit(1)

    logger.info(f"Found {len(class_dirs)} classes: {class_dirs}")

    total_maps = 0
    total_videos = 0

    for cls in class_dirs:
        cls_in = os.path.join(input_root, cls)
        cls_out = os.path.join(output_root, cls)

        video_folders = sorted(
            d for d in os.listdir(cls_in)
            if os.path.isdir(os.path.join(cls_in, d))
        )

        cls_maps = 0
        for vf in video_folders:
            n = process_video_folder(
                video_folder=os.path.join(cls_in, vf),
                out_folder=os.path.join(cls_out, vf),
            )
            cls_maps += n
            total_videos += 1

        total_maps += cls_maps
        logger.info(f"  [{cls}] {len(video_folders)} videos | {cls_maps} flow maps saved")

    logger.info(f"Done. Videos={total_videos} | Total flow maps={total_maps}")


if __name__ == "__main__":
    args = get_args()
    process_optical_flow(input_root=args.input, output_root=args.output)

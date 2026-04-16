"""
optical_flow.py
---------------
Compute dense Farneback optical flow from PNG/JPG image sequences.

Supports two input layouts automatically:

  Layout A — flat (frames directly in class folder):
    INPUT/
        Abuse/
            frame001.png
            frame002.png
        Arrest/
            ...

  Layout B — nested (UCF-Crime style, frames inside video sub-folders):
    INPUT/
        Abuse/
            Abuse001/
                frame001.png
            Abuse002/
                ...

The script detects the layout per class folder and handles both.

Output mirrors the input structure with .npy flow maps:
    OUTPUT/
        Abuse/
            frame001_flow.npy     # Layout A
        ─ or ─
        Abuse/
            Abuse001/
                frame001_flow.npy # Layout B

Usage:
    python scripts/optical_flow.py --input <frames_root> --output <flow_root>

Example:
    python scripts/optical_flow.py \\
        --input  /path/to/UCF_Crime_Frames \\
        --output /path/to/flow
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
        description="Compute Farneback optical flow from PNG/JPG image sequences."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root directory of image sequences (CLASS/*.png or CLASS/VIDEO/*.png).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root directory to write .npy flow maps (mirrors input structure).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_frames(folder: str) -> list[tuple[str, np.ndarray]]:
    """
    Load all PNG/JPG images from `folder` in sorted filename order.

    Returns:
        [(filename, bgr_array), ...]  — empty list if no images found.
    """
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    frames = []
    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            logger.warning(f"Skipping unreadable image: {os.path.join(folder, fname)}")
            continue
        frames.append((fname, img))
    return frames


def compute_optical_flow(frames: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """
    Compute Farneback dense flow between every consecutive frame pair.

    Args:
        frames: [(filename, bgr_array), ...] in temporal order.

    Returns:
        [(src_stem, flow_array), ...]  flow_array.shape == (H, W, 2).
    """
    results = []
    if len(frames) < 2:
        return results

    prev_gray = cv2.cvtColor(frames[0][1], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i][1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FB_PARAMS)
        src_stem = os.path.splitext(frames[i - 1][0])[0]
        results.append((src_stem, flow))
        prev_gray = curr_gray

    return results


def process_sequence(frame_folder: str, out_folder: str) -> int:
    """
    Load frames from `frame_folder`, compute flow, save .npy files to `out_folder`.

    Returns:
        Number of flow maps saved.
    """
    frames = load_frames(frame_folder)

    if len(frames) < 2:
        if frames:
            logger.warning(f"Need ≥2 frames for flow, skipping: {frame_folder}")
        else:
            logger.warning(f"No images found, skipping: {frame_folder}")
        return 0

    flow_pairs = compute_optical_flow(frames)
    os.makedirs(out_folder, exist_ok=True)

    for src_stem, flow in flow_pairs:
        np.save(os.path.join(out_folder, f"{src_stem}_flow.npy"), flow)

    return len(flow_pairs)


# ---------------------------------------------------------------------------
# Layout detection + main walker
# ---------------------------------------------------------------------------

def _has_images(folder: str) -> bool:
    return any(
        os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        for f in os.listdir(folder)
    )


def _has_subfolders(folder: str) -> bool:
    return any(
        os.path.isdir(os.path.join(folder, d))
        for d in os.listdir(folder)
    )


def process_class(cls_in: str, cls_out: str) -> tuple[int, int]:
    """
    Auto-detect layout and process one class directory.

    Returns:
        (video_count, flow_map_count)
    """
    total_vids = 0
    total_maps = 0

    if _has_subfolders(cls_in):
        # Layout B: CLASS/VIDEO_FOLDER/frame.png
        video_folders = sorted(
            d for d in os.listdir(cls_in)
            if os.path.isdir(os.path.join(cls_in, d))
        )
        for vf in video_folders:
            n = process_sequence(
                frame_folder=os.path.join(cls_in, vf),
                out_folder=os.path.join(cls_out, vf),
            )
            total_maps += n
            total_vids += 1

    elif _has_images(cls_in):
        # Layout A: CLASS/frame.png  (all frames treated as one sequence)
        n = process_sequence(frame_folder=cls_in, out_folder=cls_out)
        total_maps += n
        total_vids += 1

    else:
        logger.warning(f"No images or sub-folders found in: {cls_in}")

    return total_vids, total_maps


def process_optical_flow(input_root: str, output_root: str) -> None:
    """
    Walk all class directories under `input_root` and compute optical flow.
    Handles both flat (CLASS/*.png) and nested (CLASS/VIDEO/*.png) layouts.
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
        logger.error(f"No class sub-directories found in: {input_root}")
        sys.exit(1)

    logger.info(f"Found {len(class_dirs)} classes: {class_dirs}")

    total_vids = 0
    total_maps = 0

    for cls in class_dirs:
        vids, maps = process_class(
            cls_in=os.path.join(input_root, cls),
            cls_out=os.path.join(output_root, cls),
        )
        total_vids += vids
        total_maps += maps
        logger.info(f"  [{cls}] sequences={vids} | flow maps={maps}")

    logger.info(f"Done. Total sequences={total_vids} | Total flow maps={total_maps}")


if __name__ == "__main__":
    args = get_args()
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    process_optical_flow(input_root=INPUT_PATH, output_root=OUTPUT_PATH)

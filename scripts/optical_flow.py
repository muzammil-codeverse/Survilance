"""
optical_flow.py
---------------
Compute dense Farneback optical flow between consecutive extracted frames.

Usage:
    python scripts/optical_flow.py --input <frames_root> --output <flow_root>

Example:
    python scripts/optical_flow.py \
        --input  /content/frames \
        --output /content/flow
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger, log_stage

logger = get_logger("optical_flow")

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
        description="Compute Farneback optical flow from extracted frames."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root directory of extracted frames (output of extract_frames.py).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root directory to write .npy flow maps into.",
    )
    return parser.parse_args()


def _group_frames_by_video(frame_dir: str) -> dict:
    """Return {video_stem: [sorted frame paths]} from a class frame directory."""
    pattern = re.compile(r"^(.+)_frame_(\d+)\.jpg$")
    groups = defaultdict(list)

    for fname in os.listdir(frame_dir):
        m = pattern.match(fname)
        if m:
            groups[m.group(1)].append(os.path.join(frame_dir, fname))

    for stem in groups:
        groups[stem].sort(key=lambda p: int(re.search(r"_frame_(\d+)", p).group(1)))

    return groups


def compute_flow_for_class(frames_dir: str, flow_dir: str) -> int:
    """
    Compute optical flow for every consecutive frame pair in `frames_dir`.

    Returns:
        Number of flow maps saved.
    """
    os.makedirs(flow_dir, exist_ok=True)
    groups = _group_frames_by_video(frames_dir)

    if not groups:
        logger.warning(f"No grouped frames in: {frames_dir}")
        return 0

    saved = 0

    for stem, frame_paths in groups.items():
        if len(frame_paths) < 2:
            continue

        prev_gray = cv2.cvtColor(cv2.imread(frame_paths[0]), cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frame_paths)):
            curr_bgr = cv2.imread(frame_paths[i])
            if curr_bgr is None:
                logger.warning(f"Could not read frame: {frame_paths[i]}")
                continue

            curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FB_PARAMS)

            prev_stem = os.path.splitext(os.path.basename(frame_paths[i - 1]))[0]
            np.save(os.path.join(flow_dir, f"{prev_stem}_flow.npy"), flow)

            prev_gray = curr_gray
            saved += 1

    return saved


def process_optical_flow(input_root: str, output_root: str) -> None:
    """Walk all class directories under `input_root` and produce flow maps."""
    log_stage(logger, "Optical Flow Computation", f"src={input_root}  dst={output_root}")

    if not os.path.isdir(input_root):
        logger.error(f"Frames directory not found: {input_root}")
        sys.exit(1)

    class_dirs = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    total_maps = 0

    for cls in class_dirs:
        n = compute_flow_for_class(
            os.path.join(input_root, cls),
            os.path.join(output_root, cls),
        )
        total_maps += n
        logger.info(f"  [{cls}] flow maps saved: {n}")

    logger.info(f"Done. Total flow maps: {total_maps}")


if __name__ == "__main__":
    args = get_args()
    process_optical_flow(input_root=args.input, output_root=args.output)

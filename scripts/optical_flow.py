"""
optical_flow.py
---------------
Reads consecutive extracted frames for each class, computes dense Farneback
optical flow between frame_t and frame_t+1, and saves the flow maps as .npy
files (shape: H x W x 2, dtype float32).

Expected input layout  (output of extract_frames.py):
    datasets/processed/frames/
        Abuse/
            video1_frame_000000.jpg
            video1_frame_000005.jpg
            ...

Output layout:
    datasets/optical_flow/
        Abuse/
            video1_frame_000000_flow.npy
            ...
"""

import os
import sys
import re
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import PROCESSED_PATH, FLOW_OUTPUT_PATH
from utils.logger import get_logger, log_stage

logger = get_logger("optical_flow")

FRAMES_INPUT = os.path.join(PROCESSED_PATH, "frames")

# Farneback parameters (tuned for 224x224 frames)
FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def _group_frames_by_video(frame_dir: str) -> dict:
    """
    Return {video_stem: [sorted frame paths]} from a flat frame directory.
    Frame filenames are expected to end with _frame_NNNNNN.jpg
    """
    pattern = re.compile(r"^(.+)_frame_(\d+)\.jpg$")
    groups = defaultdict(list)

    for fname in os.listdir(frame_dir):
        m = pattern.match(fname)
        if m:
            groups[m.group(1)].append(os.path.join(frame_dir, fname))

    # Sort each group by frame index
    for stem in groups:
        groups[stem].sort(key=lambda p: int(re.search(r"_frame_(\d+)", p).group(1)))

    return groups


def compute_flow_for_class(frames_dir: str, flow_dir: str) -> int:
    """
    For every consecutive frame pair in `frames_dir`, compute optical flow
    and save the result to `flow_dir`.

    Returns:
        Number of flow maps saved.
    """
    os.makedirs(flow_dir, exist_ok=True)
    groups = _group_frames_by_video(frames_dir)

    if not groups:
        logger.warning(f"No grouped frames found in: {frames_dir}")
        return 0

    saved = 0

    for stem, frame_paths in groups.items():
        if len(frame_paths) < 2:
            continue  # Need at least 2 frames for a flow pair

        prev_gray = cv2.cvtColor(cv2.imread(frame_paths[0]), cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frame_paths)):
            curr_bgr = cv2.imread(frame_paths[i])
            if curr_bgr is None:
                logger.warning(f"Could not read frame: {frame_paths[i]}")
                continue

            curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, **FB_PARAMS
            )  # shape: (H, W, 2)  dtype: float32

            # Derive output filename from the *previous* frame name
            prev_stem = os.path.splitext(os.path.basename(frame_paths[i - 1]))[0]
            out_path = os.path.join(flow_dir, f"{prev_stem}_flow.npy")
            np.save(out_path, flow)

            prev_gray = curr_gray
            saved += 1

    return saved


def process_optical_flow(frames_root: str, flow_root: str) -> None:
    """
    Walk all class directories under `frames_root` and produce flow maps.
    """
    log_stage(logger, "Optical Flow Computation", f"src={frames_root}  dst={flow_root}")

    if not os.path.isdir(frames_root):
        logger.error(f"Frames directory not found: {frames_root}")
        sys.exit(1)

    class_dirs = sorted(
        d for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d))
    )

    total_maps = 0

    for cls in class_dirs:
        cls_frames = os.path.join(frames_root, cls)
        cls_flow = os.path.join(flow_root, cls)
        n = compute_flow_for_class(cls_frames, cls_flow)
        total_maps += n
        logger.info(f"  [{cls}] flow maps saved: {n}")

    logger.info(f"Done. Total flow maps: {total_maps}")


if __name__ == "__main__":
    process_optical_flow(FRAMES_INPUT, FLOW_OUTPUT_PATH)

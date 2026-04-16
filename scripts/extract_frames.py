"""
extract_frames.py
-----------------
Recursively walks a root video directory, extracts frames from every video
file using frame-skipping, resizes to IMG_SIZE, and saves them as JPEGs into
a mirrored per-class output folder.

Expected input layout:
    datasets/ucf_crime/
        Abuse/
            video1.mp4
            video2.avi
        Arrest/
            ...

Output layout:
    datasets/processed/frames/
        Abuse/
            video1_frame_0000.jpg
            video1_frame_0005.jpg
            ...
        Arrest/
            ...
"""

import os
import sys
import cv2

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import DATASET_PATH, IMG_SIZE, FRAME_SKIP, PROCESSED_PATH
from utils.logger import get_logger, log_stage

logger = get_logger("extract_frames")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
FRAMES_OUTPUT = os.path.join(PROCESSED_PATH, "frames")


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_skip: int = FRAME_SKIP,
    size: tuple = IMG_SIZE,
) -> int:
    """
    Stream through a single video, save every `frame_skip`-th frame.

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
            frame = cv2.resize(frame, size)
            filename = f"{stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def process_dataset(dataset_root: str, output_root: str) -> None:
    """
    Walk `dataset_root`, detect class folders, and extract frames for every
    video found inside each class folder.
    """
    log_stage(logger, "Frame Extraction", f"src={dataset_root}  dst={output_root}")

    if not os.path.isdir(dataset_root):
        logger.error(f"Dataset path not found: {dataset_root}")
        sys.exit(1)

    class_dirs = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )

    if not class_dirs:
        logger.error("No class sub-directories found.")
        sys.exit(1)

    logger.info(f"Found {len(class_dirs)} classes: {class_dirs}")

    total_frames = 0
    total_videos = 0

    for cls in class_dirs:
        cls_src = os.path.join(dataset_root, cls)
        cls_dst = os.path.join(output_root, cls)

        # Walk recursively to support nested layouts
        for dirpath, _, filenames in os.walk(cls_src):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VIDEO_EXTENSIONS:
                    continue

                video_path = os.path.join(dirpath, fname)
                saved = extract_frames_from_video(video_path, cls_dst)
                total_frames += saved
                total_videos += 1
                logger.info(f"  [{cls}] {fname} → {saved} frames saved")

    logger.info(f"Done. Videos processed: {total_videos} | Frames saved: {total_frames}")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, FRAMES_OUTPUT)

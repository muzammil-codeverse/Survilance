"""
preprocessing.py
----------------
SUPERSEDED by scripts/optical_flow.py for the current pipeline.

The active pipeline uses scripts/optical_flow.py which:
  - Reads pre-extracted PNG/JPG frame sequences (no VideoCapture)
  - Supports both flat (CLASS/*.png) and nested (CLASS/VIDEO/*.png) layouts
  - Is fully CLI-driven via --input / --output

This module is retained only as a reference for raw-video datasets.
"""

import cv2


def extract_frames(video_path, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames.append(frame)

    cap.release()
    return frames

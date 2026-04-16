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

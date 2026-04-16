import numpy as np


def predict_frames(model, frames):
    frames = np.array(frames) / 255.0
    predictions = model.predict(frames)
    return predictions

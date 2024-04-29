import cv2
import numpy as np

def get_video_frames(video_path: str, grayscale: bool = True) -> np.array:
    """
    Read video frames from a video file and return them as a numpy array.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    grayscale : bool
        Whether to read the frames as grayscale images.

    Returns
    -------
    np.array
        Video frames as a numpy array of shape (N, H, W, C).
    """

    cap = cv2.VideoCapture(video_path)

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()
    frames = np.array(frames)
    return frames
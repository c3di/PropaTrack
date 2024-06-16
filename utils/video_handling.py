import os
from typing import Tuple

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
    np.ndarray
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


def setup_experiment(video_name: str, result_path: str) -> Tuple[cv2.VideoWriter, np.ndarray]:
    """
    Set up the experiment by reading the video frames and creating a video writer object.

    Parameters
    ----------
    video_name : str
        Name of the video file.

    result_path : str
        Path to the result directory.

    Returns
    -------
    Tuple[cv2.VideoWriter, np.ndarray]
        Tuple containing the video writer object and the video frames.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    print(f"Generating flow for {video_name}")
    video_path = "videos/" + video_name
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames = get_video_frames(video_path)
    if not os.path.isdir(result_path):
        print(f"Directory {result_path} does not exist yet. Creating it.")
        os.mkdir(result_path)
    video_writer = cv2.VideoWriter(result_path + video_name, fourcc, 10.0, (width, height))

    return video_writer, frames

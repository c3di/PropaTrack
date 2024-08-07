"""
Pipeline takes in a video or a set of frames and processes them
to generate a vector field for reaction speed.
"""

import numpy as np
from tqdm import tqdm

from src.utils.frame_processing import (
    contours_from_front,
    front_from_frames,
    orient_normal,
    spline_from_contour,
    vec_to_nearest,
)


def length(arr: np.ndarray) -> float:
    """Calculate the length of a vector."""
    return np.linalg.norm(arr, 2)


def pipeline(frames: np.ndarray, threshold: int = 25) -> np.ndarray:
    """
    Process the frames to generate a vector field indicating the reaction speed
    at evenly spread points for each frame.

    Parameters
    ----------
    frames : np.ndarray
        Video frames as a numpy array of shape (N, H, W).

    threshold : int
        During preprocessing set all pixel values below this threshold to 0.

    Returns
    -------
    np.ndarray
        Vector field for reaction speed.

    Notes
    -------
    Format of the vector field: \n
    [frame, contour, x_pos, y_pos, x_normal, y_normal, speed]

    See Also
    --------
    video_handling.get_video_frames : Generation of frames.
    frame_processing.front_from_frames : How preprocessing is done.
    frame_processing.contours_from_front : How contours are extracted.
    """

    speeds = []

    with tqdm(
        total=len(frames) - 2, desc="Running image pipeline", colour="#6DBEA0", unit=" frames"
    ) as pbar:

        contours = contours_from_front(
            front_from_frames(frames[0], frames[1], frames[2], threshold=threshold)
        )

        i = 1
        while i < (len(frames) - 2):
            contours_next = contours_from_front(
                front_from_frames(frames[i], frames[i + 1], frames[i + 2], threshold=threshold)
            )

            for j, contour in enumerate(contours):

                for point, normal in zip(*spline_from_contour(contour)):

                    vec_nearest = vec_to_nearest(point, contours_next)

                    if vec_nearest is None:
                        continue

                    normal = orient_normal(normal, vec_nearest)

                    speeds.append(
                        [i, j, point[0], point[1], normal[0], normal[1], length(vec_nearest)]
                    )

            i += 1
            contours = contours_next
            pbar.update()

    return np.array(speeds)


def write_data(speeds: np.ndarray, result_path: str):
    """
    Write the array representing the vector field to a csv file.

    Parameters
    ----------
    speeds : np.ndarray
        Vector field for reaction speed.

    result_path : str
        Path to the result directory.

    Notes
    -------
    Format of the vector field: \n
    [frame, contour, x_pos, y_pos, x_normal, y_normal, speed]

    See Also
    --------
    pipeline : How the vector field is generated.
    """
    header = "frame,contour,x_pos,y_pos,x_normal,y_normal,speed"
    # First four columns are integers, last three are floats with 3 decimal places.
    fmt = "%d, %d, %d, %d, %1.3f, %1.3f, %1.3f"
    np.savetxt(result_path, speeds, fmt=fmt, delimiter=",", header=header)

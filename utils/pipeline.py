"""Pipeline takes in a video or a set of frames and processes them to generate a vector field for reaction speed."""
import numpy as np

from scipy.spatial.distance import cdist
from tqdm import tqdm

from utils.frame_processing import front_from_frames, contours_from_front, process_contour, spline_from_contour


def pipeline(frames: np.ndarray,
             threshold: int = 25,
             min_length: int = 5) -> np.ndarray:
    """
    Process the frames to generate a vector field indicating the reaction speed at evenly spread points for each frame.

    Parameters
    ----------
    frames : np.ndarray
        Video frames as a numpy array of shape (N, H, W).

    threshold : int
        During preprocessing set all pixel values below this threshold to 0.

    min_length : int
        Minimum length of a contour to be considered.

    Returns
    -------
    np.ndarray
        Vector field for reaction speed. Each row contains the following information:
        [frame_index, contour_index, x_position, y_position, x_normal, y_normal, speed]

    See Also
    --------
    video_handling.get_video_frames : Generation of frames.
    frame_processing.front_from_frames : How preprocessing is done.
    frame_processing.contours_from_front : How contours are extracted.
    """

    h, w = frames[0].shape
    Speeds = []

    for i in tqdm(range(len(frames[:-2])),
                  desc="Running image pipeline",
                  colour="#6DBEA0",
                  unit=" frames"):

        front = front_from_frames(frames[i],
                                  frames[i+1],
                                  threshold=threshold,
                                  version="v2")

        front_next = front_from_frames(frames[i+1],
                                       frames[i+2],
                                       threshold=threshold,
                                       version="v2")

        contours = contours_from_front(front, min_length=min_length)
        contours_next = contours_from_front(front_next, min_length=min_length)

        for j, contour in enumerate(contours):

            contour = process_contour(contour)

            spline, normals = spline_from_contour(contour)

            for spline_point, spline_normal in zip(spline, normals):
                # Distance can't be larger than image diagonal
                min_dist = h**2 + w**2
                for contour_next in contours_next:
                    distances = cdist(np.expand_dims(spline_point, axis=0), contour_next)
                    dist = np.min(distances)
                    if dist < min_dist:
                        min_dist = dist

                Speeds.append([i, j, spline_point[0], spline_point[1], spline_normal[0], spline_normal[1], min_dist])

    return np.array(Speeds)

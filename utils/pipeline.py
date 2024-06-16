"""Pipeline takes in a video or a set of frames and processes them to generate a vector field for reaction speed."""
import numpy as np

from tqdm import tqdm

from utils.frame_processing import front_from_frames, contours_from_front, process_contour, spline_from_contour, dist_to_nearest


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

    Speeds = []

    for i, _ in enumerate(tqdm(frames[:-2],
                               desc="Running image pipeline",
                               colour="#6DBEA0",
                               unit=" frames")):

        front = front_from_frames(frames[i],
                                  frames[i + 1],
                                  threshold=threshold,
                                  version="v2")

        front_next = front_from_frames(frames[i + 1],
                                       frames[i + 2],
                                       threshold=threshold,
                                       version="v2")

        contours = contours_from_front(front, min_length=min_length)
        contours_next = contours_from_front(front_next, min_length=min_length)

        for j, contour in enumerate(contours):

            contour = process_contour(contour)

            spline, normals = spline_from_contour(contour)

            for point, normal in zip(spline, normals):

                min_dist = dist_to_nearest(point, contours_next)

                if min_dist < np.inf:
                    Speeds.append([i, j, point[0], point[1], normal[0], normal[1], min_dist])

    return np.array(Speeds)


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
    fmt = "%d", "%d", "%d", "%d", "%1.3f", "%1.3f", "%1.3f"
    np.savetxt(result_path,
               speeds,
               fmt=fmt,
               delimiter=",",
               header=header)

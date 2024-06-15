"""Module contains utils for processing video frames with skimage and numpy."""
from typing import Tuple

import cv2
import numpy as np
from scipy import interpolate
from scipy.spatial.distance import cdist
from skimage import morphology


def front_from_frames(frame0: np.ndarray,
                      frame1: np.ndarray,
                      threshold: float = 35,
                      version: str = "v2") -> np.ndarray:
    """
    Generate a denoised version of the reaction front from two frames.
    
    Parameters
    ----------
    frame0 : np.ndarray
        First frame.
        
    frame1 : np.ndarray
        Second frame.

    threshold : float
        Lower threshold for binarization. Set all pixel values below this threshold to 0.

    version : str
        Version of the processing pipeline.
        
    Returns
    -------
    np.ndarray
        Denoised reaction front.

    Notes
    -----
    You can get a concise overview about mathematical morphology here:
    https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html
    """

    footprint = morphology.disk(3)

    if version == "v1":
        # Calculate the difference between two frames.
        front = frame1 - frame0
        front = front.astype(np.uint8)

        # Threshold and binarize the image.
        front[front > 180] = 0
        front[front < threshold] = 0
        front[front > 0] = 255

    elif version == "v2":
        # Threshold and binarize the images.
        frame0[frame0 > threshold] = 255
        frame0[frame0 < threshold] = 0
        frame1[frame1 > threshold] = 255
        frame1[frame1 < threshold] = 0

        # Calculate the difference between two frames.
        front = frame1 - frame0

    # Apply morphological opening.
    front = morphology.binary_opening(front, footprint)

    # Apply skeletonization.
    front = morphology.skeletonize(front)
    front = np.where(front == 1, 255, 0).astype(np.uint8)

    return front


def contours_from_front(front: np.ndarray, min_length: int = 25):
    """
    Extract contours from a reaction front.

    Parameters
    ----------
    front : np.ndarray
        Reaction front.

    min_length : int
        Minimum length of a contour.

    Returns
    -------
    np.ndarray
        Contours of the reaction front.
    """

    contours, _ = cv2.findContours(front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [np.squeeze(contour) for contour in contours if cv2.arcLength(contour, False) > min_length]

    return contours


def process_contour(contour: np.ndarray):
    """
    Process a contour by removing duplicate points and outliers.

    Parameters
    ----------
    contour : np.ndarray
        Contour to be processed.

    Returns
    -------
    np.ndarray
        Processed contour.
    """

    # Remove duplicate points.
    _, indices_unique = np.unique(contour, axis=0, return_index=True)
    contour = contour[np.sort(indices_unique), :]

    # Sample the contour at evenly spaced indices.
    length_per_arrow = 35
    num_arrows = cv2.arcLength(contour, False) / length_per_arrow + 1
    steps = max(int(len(contour) / num_arrows), 1)
    if (len(contour) - 1) % steps == 0:
        contour = contour[::steps]
    else:
        # Make sure the last point is included, so contours don't get cut off.
        contour = np.append(contour[::steps], contour[-1:], axis=0)

    intra_dists = np.diagonal(cdist(contour[:-1], contour[1:]))
    mean_dist = np.mean(intra_dists)

    outlier_indices = np.where(intra_dists > 3 * mean_dist)[0]
    if len(outlier_indices) >= 1:
        idx_first_outlier = outlier_indices[0]
        if idx_first_outlier >= len(intra_dists) - 3:
            # outliers = contour[idx_first_outlier+1:]
            # contour = contour[:idx_first_outlier]
            # for ol in outliers:
            #     ol = np.expand_dims(ol, axis=0)
            #     idx_ol = np.argmin(cdist(ol, contour))
            #     contour = np.insert(contour, idx_ol+1, ol, axis=0)
            contour = contour[:-(len(intra_dists) - idx_first_outlier)]
        else:
            contour_truncated = contour[:idx_first_outlier + 1]
            idx_last_outlier = outlier_indices[-1]
            contour_rest = contour[idx_last_outlier + 1:]
            contour = np.concatenate((contour_rest[::-1], contour_truncated))

    return contour


def calculate_distance(points):
    return sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points)))


def spline_from_contour(contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit spline to contour and calculate normals.

    Parameters
    ----------
    contour : np.ndarray
        preprocessed contour.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the spline and the normals.

    Notes
    -----
    It is advised but not necessary to preprocess the contour before applying this function.

    See Also
    --------
    process_contour: A function to preprocess the contour.
    """

    X = contour[:, 0]
    Y = contour[:, 1]

    k = 1 if len(contour) < 10 else 3

    tck, u = interpolate.splprep([X, Y], s=2, k=k)
    spline = np.array(interpolate.splev(u, tck)).T

    tangents = interpolate.splev(u, tck, der=1)
    normals = np.array([-tangents[1], tangents[0]]).T
    normals = normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)

    return spline, normals


def two_opt(points, improvement_threshold):
    count = 0
    while True:
        count += 1
        distance = calculate_distance(points)
        for i in range(len(points) - 1):
            for j in range(i + 2, len(points)):
                if j - i == 1: continue
                new_points = points[:]
                new_points[i:j] = points[i:j][::-1]
                new_distance = calculate_distance(new_points)
                if new_distance < distance:
                    points = new_points
                    break
            else:
                continue
            break
        else:
            if count > improvement_threshold:
                break
    return points

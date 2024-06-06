"""Module contains utils for processing video frames with skimage and numpy."""
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage import morphology


def front_from_frames(frame0: np.ndarray,
                      frame1: np.ndarray,
                      threshold_low: float = 35,
                      version: str = "v2") -> np.ndarray:
    """
    Generate a denoised version of the reaction front from two frames.
    
    Parameters
    ----------
    frame0 : np.ndarray
        First frame.
        
    frame1 : np.ndarray
        Second frame.

    threshold_low : float
        Lower threshold for binarization.

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
        front[front < threshold_low] = 0
        front[front > 0] = 255

    elif version == "v2":
        # Threshold and binarize the images.
        frame0[frame0 > threshold_low] = 255
        frame0[frame0 < threshold_low] = 0
        frame1[frame1 > threshold_low] = 255
        frame1[frame1 < threshold_low] = 0

        # Calculate the difference between two frames.
        front = frame1 - frame0

    # Apply morphological opening.
    front = morphology.binary_opening(front, footprint)

    # Apply skeletonization.
    front = morphology.skeletonize(front)
    front = np.where(front == 1, 255, 0).astype(np.uint8)

    return front


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

    outlier_dists = np.where(intra_dists > 3 * mean_dist)
    if len(outlier_dists[0]) >= 1:
        if outlier_dists[0][0] >= len(intra_dists) - 3:
            contour = contour[:-(len(intra_dists) - outlier_dists[0][0])]
        else:
            contour_truncated = contour[:outlier_dists[0][0] + 1]
            contour_rest = contour[outlier_dists[0][-1] + 1:]
            contour = np.concatenate((contour_rest[::-1], contour_truncated))

    return contour


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


def calculate_distance(points):
    return sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points)))


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

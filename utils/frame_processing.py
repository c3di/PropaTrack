"""Module contains utils for processing video frames with skimage and numpy."""
import cv2
from skimage import morphology
import numpy as np
from scipy.spatial.distance import cdist


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
    contour = contour[::steps]

    # Remove outliers.
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

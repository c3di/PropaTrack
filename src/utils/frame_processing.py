"""Module contains utils for processing video frames with skimage and numpy."""

import cv2
import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist
from skimage import morphology

EDGE_KERNEL = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]])

DISK_1 = morphology.disk(1)
DISK_3 = morphology.disk(3)


def front_from_frames(
    frame0: np.ndarray, frame1: np.ndarray, frame2: np.ndarray, threshold: float = 25
) -> np.ndarray:
    """
    Generate a denoised version of the reaction front from three frames.
    Front is located on frame1.

    Parameters
    ----------
    frame0, frame1, frame2 : np.ndarray
        Video frame.

    threshold : float
        Lower threshold for binarization. Set all pixel values below this threshold to 0.

    Returns
    -------
    np.ndarray
        Denoised reaction front.

    Notes
    -----
    You can get a concise overview about mathematical morphology here:
    https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html
    """

    _binarize(frame0, threshold)
    _binarize(frame1, threshold)
    _binarize(frame2, threshold)

    edges0 = _edges_from_frame(frame0)
    edges1 = _edges_from_frame(frame1)
    edges2 = _edges_from_frame(frame2)

    front = _front_from_edges(edges0, edges1, edges2)

    front = _apply_morphology(front)

    return front


def contours_from_front(front: np.ndarray, min_length: int = 5) -> list:
    """
    Extract contours from a reaction front.

    Parameters
    ----------
    front : np.ndarray
        Reaction front.

    min_length : int
        Minimum length of a contour to be considered.

    Returns
    -------
    list
        Contours of the reaction front.
    """

    contours, _ = cv2.findContours(front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [
        np.squeeze(contour) for contour in contours if cv2.arcLength(contour, False) > min_length
    ]

    return contours


def process_contour(contour: np.ndarray, sampling_factor: int) -> np.ndarray:
    """
    Process a contour by removing duplicate points and outliers and downsampling.
    """

    contour = _remove_duplicates(contour)

    contour = _sample_contour(contour, sampling_factor)

    return contour


def nearest_point(point: np.ndarray, contours: list[np.ndarray]) -> np.ndarray | None:
    """Find the nearest point on a list of contours to a given point."""
    if contours:
        idx_contour = 0
        idx_point = 0
        min_dist = np.inf
        for idx, contour in enumerate(contours):
            distances = cdist(np.expand_dims(point, axis=0), contour)
            dist_idx = np.argmin(distances)
            dist = distances[0, dist_idx]
            if dist < min_dist:
                min_dist = dist
                idx_contour = idx
                idx_point = dist_idx
        point = contours[idx_contour][idx_point]
        return point
    return None


def _binarize(frame: np.ndarray, threshold: float = 25) -> None:
    """Binarize the frames by setting all pixel values below a threshold to 0."""
    frame[frame > threshold] = 255
    frame[frame < threshold] = 0


def _edges_from_frame(frame: np.ndarray) -> np.ndarray:
    """Extract edges from a binarized frame using a simple derivative filter."""
    edges = np.abs(signal.convolve2d(frame, EDGE_KERNEL, mode="same", boundary="symm")).astype(
        np.uint8
    )
    _binarize(edges)
    return edges


def _front_from_edges(edges0: np.ndarray, edges1: np.ndarray, edges2: np.ndarray) -> np.ndarray:
    """
    Get the reaction front from the edges detected in three frames.
    Front is located in the frame corresponding to edges1.
    """
    edges0 = cv2.dilate(edges0, DISK_1, iterations=3)
    edges2 = cv2.dilate(edges2, DISK_1, iterations=3)
    front = edges1 - edges0 - edges2
    front[front != 255] = 0
    return front


def _apply_morphology(front: np.ndarray) -> np.ndarray:
    """Apply morphological operations to the reaction front."""

    front = morphology.closing(front, DISK_3)
    front = morphology.skeletonize(front)

    front = np.where(front == 1, 255, 0).astype(np.uint8)

    return front


def _remove_duplicates(contour: np.ndarray) -> np.ndarray:
    """Remove duplicate points from a contour."""
    _, indices_unique = np.unique(contour, axis=0, return_index=True)
    contour = contour[np.sort(indices_unique), :]
    return contour


def _sample_contour(contour: np.ndarray, sampling_factor: int) -> np.ndarray:
    """Sample the contour to get evenly spaced points."""
    num_arrows = cv2.arcLength(contour, False) / sampling_factor + 1
    steps = max(int(len(contour) / num_arrows), 1)
    if (len(contour) - 1) % steps == 0:
        contour = contour[::steps]
    else:
        # Make sure the last point is included, so contours don't get cut off.
        contour = np.append(contour[::steps], contour[-1:], axis=0)
    return contour

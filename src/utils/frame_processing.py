"""Module contains utils for processing video frames with skimage and numpy."""

from typing import Tuple

import cv2
import numpy as np
from scipy import interpolate, signal
from scipy.spatial.distance import cdist
from skimage import morphology

_EDGE_KERNEL = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]])

_DISK_1 = morphology.disk(1)
_DISK_3 = morphology.disk(3)


def front_from_frames(
    frame0: np.ndarray,
    frame1: np.ndarray,
    frame2: np.ndarray,
    threshold: float = 25,
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


def contours_from_front(front: np.ndarray) -> list[np.ndarray]:
    """
    Extract contours from a reaction front.

    Parameters
    ----------
    front : np.ndarray
        Reaction front.

    Returns
    -------
    list[np.ndarray]
        Contours of the reaction front.
    """

    contours, _ = cv2.findContours(front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [np.squeeze(contour) for contour in contours if cv2.arcLength(contour, False) > 10]

    return contours


def spline_from_contour(contour: np.ndarray, sample_gap: int) -> Tuple[np.ndarray, np.ndarray]:
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

    sample_gap: int
        Number of pixels between two sampling points on a contour.

    Notes
    -----
    It is advised but not necessary to preprocess the contour before applying this function.

    See Also
    --------
    _process_contour: A function to preprocess the contour.
    """

    contour = _process_contour(contour, sample_gap)

    x = contour[:, 0]
    y = contour[:, 1]

    k = 1 if len(contour) < 10 else 3

    tck, u = interpolate.splprep([x, y], s=2, k=k)
    spline = np.array(interpolate.splev(u, tck)).T

    tangents = interpolate.splev(u, tck, der=1)
    normals = np.array([-tangents[1], tangents[0]]).T
    normals = normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)

    return spline, normals


def vec_to_nearest(point: np.ndarray, contours: list[np.ndarray]) -> np.ndarray | None:
    """
    Find the vector to the nearest contour in the next frame.

    Parameters
    ----------
    point : np.ndarray
        Point on a spline given as [x, y].

    contours : list[np.ndarray]
       List of all contours in the next frame.

    Returns
    -------
    np.ndarray
        Vector to the nearest contour.

    """
    if contours:
        idx_contour = 0
        idx_point = 0
        min_dist = np.inf
        point_expanded = np.expand_dims(point, axis=0)
        for idx, contour in enumerate(contours):
            distances = cdist(point_expanded, contour)
            dist_idx = np.argmin(distances)
            dist = distances[0, dist_idx]
            if dist < min_dist:
                min_dist = dist
                idx_contour = idx
                idx_point = dist_idx
        nearest_point = contours[idx_contour][idx_point]

        return nearest_point - point

    return None


def get_direction(normal: np.ndarray, vec_nearest: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Decide which vector to use for indicating the direction of the reaction front.
    """

    speed = np.linalg.norm(vec_nearest)
    vec_nearest_normalized = vec_nearest / speed
    dir_similarity = np.dot(normal, vec_nearest_normalized)

    if abs(dir_similarity) < 0.5:
        speed *= abs(dir_similarity)
        return vec_nearest_normalized, speed
    if dir_similarity < 0:
        return -normal, speed
    return normal, speed


def _binarize(frame: np.ndarray, threshold: float = 25) -> None:
    """Binarize the frames by setting all pixel values below a threshold to 0."""
    frame[frame > threshold] = 255
    frame[frame < threshold] = 0


def _edges_from_frame(frame: np.ndarray) -> np.ndarray:
    """Extract edges from a binarized frame using a simple derivative filter."""
    edges = np.abs(signal.convolve2d(frame, _EDGE_KERNEL, mode="same", boundary="symm")).astype(
        np.uint8
    )
    _binarize(edges)
    return edges


def _front_from_edges(edges0: np.ndarray, edges1: np.ndarray, edges2: np.ndarray) -> np.ndarray:
    """
    Get the reaction front from the edges detected in three frames.
    Front is located in the frame corresponding to edges1.
    """
    edges0 = cv2.dilate(edges0, _DISK_1, iterations=3)
    edges2 = cv2.dilate(edges2, _DISK_1, iterations=3)
    front = edges1 - edges0 - edges2
    front[front != 255] = 0

    return front


def _apply_morphology(front: np.ndarray) -> np.ndarray:
    """Apply morphological operations to the reaction front."""

    front = morphology.binary_closing(front, _DISK_3)
    front = morphology.skeletonize(front)

    front = np.where(front == 1, 255, 0).astype(np.uint8)

    return front


def _remove_duplicates(contour: np.ndarray) -> np.ndarray:
    """Remove duplicate points from a contour."""
    _, indices_unique = np.unique(contour, axis=0, return_index=True)
    contour = contour[np.sort(indices_unique), :]
    return contour


def _sample_contour(contour: np.ndarray, sample_gap: int) -> np.ndarray:
    """Sample the contour to get evenly spaced points."""
    num_arrows = cv2.arcLength(contour, False) / sample_gap + 1
    steps = max(int(len(contour) / num_arrows), 1)
    if (len(contour) - 1) % steps == 0:
        contour = contour[::steps]
    else:
        # Make sure the last point is included, so contours don't get cut off.
        contour = np.append(contour[::steps], contour[-1:], axis=0)
    return contour


def _find_outliers(contour: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find indices of outliers in a contour.

    Notes
    -----
    An outlier is defined as a point with a distance to the next point
    greater than 2 times the mean distance between all points.
    """
    intra_dists = np.diagonal(cdist(contour[:-1], contour[1:]))
    mean_dist = np.round(np.mean(intra_dists), 1)
    (outlier_indices,) = np.where(intra_dists > 2 * mean_dist)
    # Add 1 to get the index of the second point in the pair.
    return outlier_indices + 1, mean_dist


def _handle_outliers(contour: np.ndarray) -> np.ndarray:
    """
    Handle outliers in a contour by removing them directly or reordering the contour.
    """
    outlier_indices, mean_dist = _find_outliers(contour)

    if outlier_indices.size > 0:
        idx_first_outlier = outlier_indices[0]
        contour_truncated = contour[:idx_first_outlier]
        idx_last_outlier = outlier_indices[-1]
        contour_rest = contour[idx_last_outlier:]
        if np.linalg.norm(contour_rest[0] - contour_truncated[0]) <= 2 * mean_dist:
            contour_rest_rev = contour_rest[::-1]
            contour = np.concatenate((contour_rest_rev, contour_truncated))
        else:
            if contour_rest.size > contour_truncated.size:
                return contour_rest
            return contour_truncated

    return contour


def _process_contour(contour: np.ndarray, sample_gap: int) -> np.ndarray:
    """
    Process a contour by removing duplicate points and outliers and downsampling.
    """
    contour = _remove_duplicates(contour)

    contour = _sample_contour(contour, sample_gap)

    contour = _handle_outliers(contour)

    return contour

"""Module contains utils for processing video frames with skimage and numpy."""
from skimage import morphology
import numpy as np


def front_from_frames(frame0: np.ndarray,
                      frame1: np.ndarray,
                      footprint: np.ndarray,
                      threshold_low: float = 20) -> np.ndarray:
    """
    Generate a denoised version of the reaction front from two frames.
    
    Parameters
    ----------
    frame0 : np.ndarray
        First frame.
        
    frame1 : np.ndarray
        Second frame.
        
    footprint : np.ndarray
        Structuring element for morphological operations.

    threshold_low : float
        Lower threshold for binarization.
        
    Returns
    -------
    np.ndarray
        Denoised reaction front.

    Notes
    -----
    You can get a concise overview about mathematical morphology here:
    https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html
    """

    # Calculate the difference between two frames.
    front = frame1 - frame0
    front = front.astype(np.uint8)

    # Threshold and binarize the image.
    front[front > 180] = 0
    front[front < threshold_low] = 0
    front[front > 0] = 255

    # Apply morphological opening.
    front = morphology.binary_opening(front, footprint)

    # Apply skeletonization.
    front = morphology.skeletonize(front)
    front = np.where(front == 1, 255, 0).astype(np.uint8)

    return front


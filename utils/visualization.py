import os.path

import cv2
from matplotlib import pyplot as plt
from skimage import morphology
import numpy as np
from scipy import interpolate

from utils.video_handling import get_video_frames

IMG_DIR = "results/visualization"


def plot_simple(img: np.ndarray, title: str, img_dir: str, cmap: str = "gray") -> None:
    """Plot a single image."""
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.savefig(f"{img_dir}/{title}.png")
    plt.close()


def plot_pipeline_steps(video_path: str, frame0: int, frame1: int, img_dir: str = IMG_DIR) -> None:
    """
    Plot the pipeline steps for the reaction front detection.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame0 : int
        Index of the first frame.
    frame1 : int
        Index of the second frame.
    img_dir : str
        Directory for saving the images.
    """

    cmap = plt.get_cmap("tab20")

    if not os.path.isdir(img_dir):
        print("Directory does not exist. Creating it now.")
        os.makedirs(img_dir, exist_ok=True)

    footprint = morphology.disk(3)
    frames = get_video_frames(video_path)

    frame_1 = frames[frame1]
    frame_0 = frames[frame0]

    plot_simple(frame_0, f"frame_{frame0}", img_dir)
    plot_simple(frame_1, f"frame_{frame1}", img_dir)

    front = frame_1 - frame_0
    front = front.astype(np.uint8)

    plot_simple(front, "front_raw", img_dir)

    front[front > 180] = 0
    front[front < 20] = 0
    front[front > 0] = 255

    plot_simple(front, "front_denoised_binarized", img_dir)

    front = morphology.binary_opening(front, footprint)

    front = morphology.medial_axis(front)
    front = np.where(front == 1, 255, 0).astype(np.uint8)

    plot_simple(front, "front_morphological", img_dir)

    h, w = front.shape
    contour_canvas = np.zeros((h, w), dtype=np.uint8)
    spline_canvas = np.zeros((h, w), dtype=np.uint8)
    arrow_canvas = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    length_per_arrow = 35
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax1.imshow(contour_canvas, cmap="gray")
    ax2.imshow(spline_canvas, cmap="gray")
    ax3.imshow(arrow_canvas, cmap="gray")

    for j, contour in enumerate(contours):
        _, indices_unique = np.unique(contours[0], axis=0, return_index=True)
        contour = np.squeeze(contours[0][np.sort(indices_unique), :])

        num_arrows = cv2.arcLength(contour, False) / length_per_arrow + 1
        steps = max(int(len(contour) / num_arrows), 1)

        contour = contour[::steps]
        X = contour[:, 0]
        Y = contour[:, 1]

        ax1.plot(X, Y, color=cmap(j))

        k = 1 if len(contour) < 10 else 3

        tck, u = interpolate.splprep([X, Y], s=2, k=3)
        out = np.array(interpolate.splev(u, tck)).T

        ax2.plot(out[:, 0], out[:, 1], color=cmap(j))
        ax3.plot(out[:, 0], out[:, 1], color=cmap(j))

        tangent = interpolate.splev(u, tck, der=1)
        normal = np.array([-tangent[1], tangent[0]]).T
        normal = normal / np.linalg.norm(normal, axis=1).reshape(-1, 1)

        plt.quiver(out[1:-1, 0], out[1:-1, 1], normal[1:-1, 0], normal[1:-1, 1],
                   color='r', angles='xy', scale_units='xy', scale=0.01, width=0.001)

    fig1.savefig(f"{img_dir}/contours.png")
    fig2.savefig(f"{img_dir}/splines.png")
    fig3.savefig(f"{img_dir}/arrows.png")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

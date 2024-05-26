import os.path

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
from skimage import morphology

from utils.frame_processing import contours_from_front, process_contour, front_from_frames
from utils.video_handling import get_video_frames

IMG_DIR = "results/visualization"


def plot_simple(img: np.ndarray, title: str, img_dir: str, cmap: str = "gray") -> None:
    """Plot a single image."""
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.savefig(f"{img_dir}/{title}.png")
    plt.close()


def plot_pipeline_steps(video_path: str,
                        frame_idx0: int,
                        frame_idx1: int,
                        img_dir: str = IMG_DIR,
                        version: str = "v2") -> None:
    """
    Plot the pipeline steps for the reaction front detection.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_idx0 : int
        Index of the first frame.
    frame_idx1 : int
        Index of the second frame.
    img_dir : str
        Directory for saving the images. Default: IMG_DIR.
    version : str
        Version of the pipeline to use. Possible values: {"v1", "v2"}.
    """
    os.makedirs(img_dir + "/" + version, exist_ok=True)
    cmap = plt.get_cmap("plasma")

    if not os.path.isdir(img_dir):
        print("Directory does not exist. Creating it now.")
        os.makedirs(img_dir, exist_ok=True)

    footprint = morphology.disk(3)
    frames = get_video_frames(video_path)

    frame_1 = frames[frame_idx1]
    frame_0 = frames[frame_idx0]

    plot_simple(frame_0, f"frame_{frame_idx0}", img_dir + "/" + version)
    plot_simple(frame_1, f"frame_{frame_idx1}", img_dir + "/" + version)

    if version == "v1":
        front = frame_1 - frame_0
        front = front.astype(np.uint8)

        plot_simple(front, "front_raw", img_dir + "/" + version)

        front[front > 180] = 0
        front[front < 20] = 0
        front[front > 0] = 255

    elif version == "v2":
        # Threshold and binarize the images.
        frame_0[frame_0 > 35] = 255
        frame_0[frame_0 < 35] = 0
        frame_1[frame_1 > 35] = 255
        frame_1[frame_1 < 35] = 0

        plot_simple(frame_0, f"frame_{frame_idx0}_denoised", img_dir + "/" + version)
        plot_simple(frame_1, f"frame_{frame_idx1}_denoised", img_dir + "/" + version)

        # Calculate the difference between two frames.
        front = frame_1 - frame_0

        plot_simple(front, "front_difference", img_dir + "/" + version)

    front = morphology.binary_opening(front, footprint)

    front = morphology.skeletonize(front)
    front = np.where(front == 1, 255, 0).astype(np.uint8)

    plot_simple(front, "front_morphological", img_dir + "/" + version)

    h, w = front.shape
    contour_canvas = np.zeros((h, w), dtype=np.uint8)
    spline_canvas = np.zeros((h, w), dtype=np.uint8)
    arrow_canvas = np.zeros((h, w), dtype=np.uint8)
    contours = contours_from_front(front)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax1.imshow(contour_canvas, cmap="gray")
    ax2.imshow(spline_canvas, cmap="gray")
    ax3.imshow(arrow_canvas, cmap="gray")

    # Need next front for arrow length calculation
    front_next = front_from_frames(frames[frame_idx1], frames[frame_idx1 + 1])
    contours_next = contours_from_front(front_next)

    for j, contour in enumerate(contours):

        min_dist = w**2 + h**2 # Distances can't be larger than image diagonal
        for contour_next in contours_next:
            distances = cdist(contour, contour_next)
            dist = np.mean(np.min(distances, axis=0))
            if dist < min_dist:
                min_dist = dist

        contour = process_contour(contour)

        X = contour[:, 0]
        Y = contour[:, 1]

        ax1.plot(X, Y, color=cmap(j))

        # Use different spline interpolation for small and large contours
        k = 1 if len(contour) < 10 else 3

        tck, u = interpolate.splprep([X, Y], s=2, k=3)
        out = np.array(interpolate.splev(u, tck)).T

        ax2.plot(out[:, 0], out[:, 1], color=cmap(j))
        ax3.plot(out[:, 0], out[:, 1], color=cmap(j))

        tangent = interpolate.splev(u, tck, der=1)
        normal = np.array([-tangent[1], tangent[0]]).T
        normal = normal / np.linalg.norm(normal, axis=1).reshape(-1, 1)

        plt.quiver(out[1:-1, 0], out[1:-1, 1], normal[1:-1, 0], normal[1:-1, 1],
                   color=cmap(int(min_dist)), angles='xy', scale_units='xy', scale=0.01, width=0.001)

    fig1.savefig(img_dir + "/" + f"{version}/contours.png")
    fig2.savefig(img_dir + "/" + f"{version}/splines.png")
    fig3.savefig(img_dir + "/" + f"{version}/arrows.png")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

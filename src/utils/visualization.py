"""Functionality for visualizing results and intermediate steps of the image pipeline."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

IMG_DIR = "results/visualization"


def plot_simple(img: np.ndarray, title: str, img_dir) -> None:
    """Plot a single image."""
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(f"{img_dir}/{title}.png", dpi=400)
    plt.close()


def plot_contour(contour: np.ndarray, canvas: np.ndarray, title: str, img_dir: str) -> None:
    """Plot a single contour."""
    fig, axis = plt.subplots()
    axis.imshow(canvas, cmap="gray")
    axis.plot(contour[:, 0], contour[:, 1], color="red")
    plt.axis("off")
    fig.savefig(f"{img_dir}/{title}.png", dpi=400)
    plt.close()


DFKI_COLORS = np.array(
    [
        [0.02352941, 0.09019608, 0.10980392],
        [0.04313726, 0.11764706, 0.20392157],
        [0.05882353, 0.14509805, 0.2901961],
        [0.07843138, 0.17254902, 0.38039216],
        [0.09411765, 0.2, 0.47058824],
        [0.11372549, 0.22745098, 0.56078434],
        [0.27450982, 0.25490198, 0.57254905],
        [0.43529412, 0.28627452, 0.58431375],
        [0.59607846, 0.31764707, 0.59607846],
        [0.7529412, 0.34509805, 0.60784316],
        [0.91764706, 0.3764706, 0.62352943],
        [0.827451, 0.4509804, 0.627451],
        [0.7254902, 0.52156866, 0.627451],
        [0.627451, 0.59607846, 0.6313726],
        [0.5254902, 0.67058825, 0.63529414],
        [0.42352942, 0.7411765, 0.63529414],
        [0.5137255, 0.7294118, 0.53333336],
        [0.62352943, 0.70980394, 0.41960785],
        [0.7372549, 0.69411767, 0.30980393],
        [0.84705883, 0.6745098, 0.19607843],
        [0.9529412, 0.654902, 0.08235294],
    ]
)

dfki_cmap = LinearSegmentedColormap.from_list("custom_cmap", DFKI_COLORS, N=256)


def plot_vector_field(results: np.ndarray, result_path: str) -> None:
    """
    Plot the vector field for the reaction speed.

    Parameters
    ----------
    results : np.ndarray
        Vector field for reaction speed.

    result_path : str
        Where to save the result.

    See Also
    ----------
    pipeline.pipeline : Format of results.
    """
    c_low_speed = "#0E2346"
    c_avg_speed = "#7D4C97"
    c_high_speed = "#EB629F"

    fig = plt.figure(dpi=400)
    avg_speed = np.mean(results[results[:, -1] < 15][:, -1])
    for result in tqdm(results, desc="Plotting vector field ", colour="#6DBEA0", unit=" vectors"):

        x, y, nx, ny, speed = result[2:]

        if speed > 15:
            continue

        color = c_avg_speed
        if speed < avg_speed - 2:
            color = c_low_speed
        elif speed > avg_speed + 2:
            color = c_high_speed

        plt.quiver(
            x,
            y,
            nx,
            ny,
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1 / (speed + 10e-6),
            width=0.001,
        )

    plt.axis("off")
    plt.savefig(result_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Create a gradient image to show the colormap
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    figure, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(gradient, aspect="auto", cmap=dfki_cmap)
    ax.set_axis_off()

    plt.show()

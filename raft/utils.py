"""Functionality for applying and visualizing the RAFT model."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.io import read_video

plt.rcParams["savefig.bbox"] = "tight"


def get_video_frames(video_path: str) -> torch.Tensor:
    """
    Read video frames from a video file and return them as a tensor.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    torch.Tensor
        Video frames as a tensor of shape (N, C, H, W).
    """
    frames, _, _ = read_video(str(video_path), pts_unit="sec")
    frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    return frames


def plot_frames(images: torch.Tensor | list[torch.Tensor], **imshow_kwargs) -> None:
    """
    Plot a grid of images. With the shape of the grid determined by the shape of the input object.
    Source: https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html

    Parameters
    ----------
    images : torch.Tensor or list[torch.Tensor]
        Images to plot. Ordering of the input determines the grid shape.

    imshow_kwargs : dict
        Additional keyword arguments to pass to `ax.imshow`.
    """
    if not isinstance(images[0], list):
        # Make a 2d grid even if there's just 1 row
        images = [images]

    num_rows = len(images)
    num_cols = len(images[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


def preprocess(batch: torch.Tensor) -> torch.Tensor:
    """
    Takes a batch of images and normalizes them to [-1, 1] and resizes them to (704, 1600).

    Parameters
    ----------
    batch : torch.Tensor
        Batch of images to preprocess of shape (N, C, H, W).

    Returns
    -------
    torch.Tensor
        Preprocessed batch of images of shape (N, C, 704, 1600).
    """
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch

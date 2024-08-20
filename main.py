"""Main script to run the pipeline generating the vector field."""

import argparse
import os

from src.utils.pipeline import pipeline, write_data
from src.utils.video_handling import get_video_frames
from src.utils.visualization import plot_vector_field


def get_paths(arguments: argparse.Namespace):
    """Get the paths for the result files."""

    # Make sure that the path has the correct format.
    video_path = os.path.normpath(arguments.video_path)
    # video_path.split("/")[-1] give the name of the video.
    result_name = video_path.split("\\")[-1].split(".")[0]
    result_path_img = f"{arguments.result_dir}/vector_field_{result_name}.png"
    result_path_txt = f"{arguments.result_dir}/{result_name}.txt"
    return result_path_img, result_path_txt


def main(arguments: argparse.Namespace) -> None:
    """Main function to run the pipeline."""

    result_path_img, result_path_txt = get_paths(arguments)

    frames = get_video_frames(arguments.video_path)
    results = pipeline(frames, arguments.threshold, arguments.sample_gap)
    write_data(results, result_path_txt)

    print(f"Successfully saved the vector field as a .txt file to {arguments.result_dir}.")
    if arguments.show:
        plot_vector_field(results, result_path_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PropaTrack",
        description="Track the speed of self-propagating"
        " exothermic reactions in reactive multilayer foils.",
    )

    parser.add_argument("video_path", help="Path to the raw video.")
    parser.add_argument("result_dir", help="Directory to save the result.")

    parser.add_argument(
        "-t",
        "--threshold",
        default=25,
        type=int,
        help="Minimum intensity for pixels to be considered part of the front. Default: 25.",
    )

    parser.add_argument(
        "-g",
        "--sample_gap",
        default=15,
        type=int,
        help="Number of pixels between two sampling points on a contour. Default: 15",
    )

    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Whether to show the vector field after running the pipeline.",
    )

    args = parser.parse_args()

    main(args)

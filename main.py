import argparse

from utils.pipeline import pipeline, write_data
from utils.video_handling import get_video_frames
from utils.visualization import plot_vector_field


def restricted_int(x):
    """Type for argparse - int between 0 and 255."""
    x = int(x)
    if x < 0 or x > 255:
        raise argparse.ArgumentTypeError("%r not in range [0, 255]" % (x,))
    return x


def get_paths(args: argparse.Namespace):
    """Get the paths for the result files."""

    # video_path.split("/")[-1] give the name of the video.
    result_name = args.video_path.split("/")[-1].split(".")[0]
    result_path_img = f"{args.result_dir}/vector_field_{result_name}.png"
    result_path_txt = f"{args.result_dir}/{result_name}.txt"
    return result_path_img, result_path_txt


def main(args: argparse.Namespace) -> None:
    """Main function to run the pipeline."""

    result_path_img, result_path_txt = get_paths(args)

    frames = get_video_frames(args.video_path)
    results = pipeline(frames, args.threshold, args.min_length)
    write_data(results, result_path_txt)

    print(f"Successfully saved the vector field as a .txt file to {args.result_dir}.")
    if args.show:
        plot_vector_field(results, result_path_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PropaTrack',
        description='Track the speed of self-propagating exothermic reactions in reactive multilayer foils.'
    )

    parser.add_argument('video_path', help='Path to the raw video.')
    parser.add_argument('result_dir', help='Directory to save the result.')

    parser.add_argument('-t',
                        '--threshold',
                        default=25,
                        type=restricted_int,
                        help='Minimum intensity for pixels to be considered part of the front. Default: 25.')
    parser.add_argument('-l',
                        '--min_length',
                        default=5,
                        help="Minimum length for contours to not be considered as noise. Default: 5.")
    parser.add_argument('-s',
                        '--show',
                        action='store_true',
                        help="Whether to show the vector field after running the pipeline.")

    args = parser.parse_args()

    main(args)

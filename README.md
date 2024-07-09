[![NumPy Badge](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=fff&style=for-the-badge)](https://github.com/numpy/numpy)
[![OpenCV Badge](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=fff&style=for-the-badge)](https://github.com/opencv/opencv)

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# PropaTrack

PropaTrack is a command line tool and soon Python library for tracking the speed of self-propagating reactions on
reactive multilayer foils. It is an image pipeline that takes in frames from a video recording of the reaction
and puts out a vector field of the reaction speed at every timestep of the reaction.

## Installation
You can set up PropaTrack by cloning the repository and installing the dependencies using poetry as follows:
```bash
git clone https://github.com/Thunfischpirat/PropaTrack
cd PropaTrack
poetry install
```

Alternatively, instead of using poetry, you can install the dependencies manually using pip:
```bash
pip install -r requirements.txt
```

## Usage
You can run PropaTrack by executing the following command:
```bash
poetry run main.py --show videos/VIDEO_NAME.mp4 results
```
Or, using your loca installation of Python:
```bash
python main.py --show videos/VIDEO_NAME.mp4 results
```

Replace `videos/VIDEO_NAME.mp4` with the path to the video you want to analyze. 
The results will be saved in the `results` directory. The `--show` flag will display the resulting vector field
in a separate window. The vector field will also be saved as a .txt file in the `results` directory.

"""
This module contains functionality for documenting experiments
executed in the jupyter notebook.
"""

import os
from time import gmtime, strftime


class Documenter:
    """Document parameters and information about an experiment."""

    def __init__(
        self,
        experiment_dir: str,
        experiment_name: str,
        experiment_params: dict,
        exists_ok: bool = False,
    ):
        """
        Set up experiment documentation in a txt file.
        """
        self.path_to_experiment = f"{experiment_dir}/{experiment_name}"
        os.makedirs(self.path_to_experiment, exist_ok=exists_ok)
        self.file_path = self.path_to_experiment + "/experiment_log.txt"

        # Write initial content
        self.write_initial_content(experiment_name, experiment_params)

    def write_initial_content(self, experiment_name: str, experiment_params: dict):
        """Write the initial content of the experiment protocol."""
        with open(self.file_path, "w", encoding="utf-8") as file:
            file.write("Experiment Protocol:\n\n")
            file.write(f"Name: {experiment_name}\n")
            file.write(f"Time of Start: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}\n\n")

            for key, value in experiment_params.items():
                file.write(f"{key}: {value}\n")

            file.write("Further information:\n")

    def comment(self, comment: str) -> None:
        """Add some extra information about the experiment such as purpose."""
        with open(self.file_path, "a", encoding="utf-8") as file:
            file.write("\nComment:\n")
            file.write(comment)
            file.write("\n\n")

    def log(self, info: str) -> None:
        """Add some more information to the experiment protocol."""
        time = strftime("%H:%M:%S", gmtime())
        with open(self.file_path, "a", encoding="utf-8") as file:
            file.write(f"{time}: {info}\n")

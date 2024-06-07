"""This module contains functionality for documenting experiments executed in the jupyter notebook."""
import os
from typing import Dict
from time import gmtime, strftime


class Documenter:
    """Document parameters and information about an experiment."""
    def __init__(self,
                 experiment_dir: str,
                 experiment_name: str,
                 experiment_params: Dict,
                 exists_ok: bool = False):
        """
        Set up experiment documentation in a txt file.
        """
        path_to_experiment = f"{experiment_dir}/{experiment_name}"
        os.makedirs(path_to_experiment, exist_ok=exists_ok)


        file_path = path_to_experiment + "/experiment_log.txt"
        self.experiment = open(file_path, "w")

        self.experiment.write("Experiment Protocol:\n\n")
        self.experiment.write(f"Name: {experiment_name}\n")
        self.experiment.write(f"Time of Start: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}\n\n")

        for key, value in experiment_params.items():
            self.experiment.write(f"{key}: {value}\n")

        self.experiment.write("Further information:\n")

    def comment(self, comment: str) -> None:
        """Add some extra information about the experiment such as purpose."""
        self.experiment.write("\n")
        self.experiment.write("Comment:\n")
        self.experiment.write(comment)
        self.experiment.write("\n\n")

    def log(self, info: str) -> None:
        """ Add some more information to the experiment protocol."""
        time = strftime('%H:%M:%S', gmtime())
        self.experiment.write(f"{time}: {info}\n")

    def close(self) -> bool:
        """Close documentation file."""
        self.experiment.close()
        return self.experiment.closed
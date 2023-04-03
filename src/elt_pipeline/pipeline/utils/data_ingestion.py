import os
import json

class DataIngestion:
    """ """
    @staticmethod
    def save_to_file(data, file_path):
        """

        :param data: 
        :param file_path: 

        """
        # Save data to a file
        with open(file_path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_from_file(file_path):
        """

        :param file_path: 

        """
        # Load data from a file
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
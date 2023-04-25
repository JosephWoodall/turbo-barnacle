'''
retrieves the source data for the pipeline and returns the data in one conoslidated location
'''
import inspect
from faker import Faker


class SourceDataRetrieval:
    def __init__(self):
        print("-----SOURCE DATA RETRIEVAL INITIALIZED-----")

    def _fake_data_generator(self, num_rows=100, num_cols=100) -> dict:
        """
        _fake_data_generator generates fake data for testing purposes only.

        Args:
            num_rows (int, optional): number of rows to be returned in the dictionary. Defaults to 100.
            num_cols (int, optional): number of columns to be returned in the dictionary. Defaults to 100.

        Returns:
            dict: dictionary of synthetically generated data using the Faker library.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.faker = Faker()
        self.data = {}

        for i in range(self.num_cols):
            self.data[f'col{i}'] = []
        for _ in range(self.num_rows):
            for i in range(self.num_cols):
                self.data[f'col{i}'].append(self.faker.random_element())
        print(list(self.data.items())[0])
        return self.data


if __name__ == '__main__':
    SourceDataRetrieval()._fake_data_generator()

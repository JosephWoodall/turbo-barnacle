import csv
from typing import List
import pytest

'''
In this example, the CsvTester class takes a list of filenames as input, loads the data from the csv files, and provides two test methods: test_column_values and test_column_names. test_column_values takes a column name as input and checks that the values in that column are the same across all files. test_column_names checks that the column names are the same across all files.

You can add additional test methods as needed to check for other differences between the files. To run the tests, you can use Pytest by calling pytest on the test file containing the test_csv_files function.
'''


class CsvTester:
    def __init__(self, filenames: List[str]):
        self.filenames = filenames
        self.data = {}
        self.load_data()

    def load_data(self):
        for filename in self.filenames:
            with open(filename, "r") as f:
                reader = csv.DictReader(f)
                self.data[filename] = list(reader)

    def test_column_values(self, column_name: str):
        unique_values = []
        for filename, rows in self.data.items():
            for row in rows:
                value = row.get(column_name)
                if value not in unique_values:
                    unique_values.append(value)
        assert len(
            unique_values) == 1, f"{column_name} column has different values across files"

    def test_column_names(self):
        column_names = None
        for filename, rows in self.data.items():
            if not column_names:
                column_names = rows[0].keys()
            else:
                assert column_names == rows[0].keys(
                ), f"Column names differ in {filename}"

    def run_tests(self):
        for column_name in self.data[self.filenames[0]][0].keys():
            self.test_column_values(column_name)
        self.test_column_names()


def test_csv_files():
    tester = CsvTester(["file1.csv", "file2.csv", "file3.csv"])
    tester.run_tests()


test_csv_files()

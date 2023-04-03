import unittest
from data_prep import cleaning


class TestCleaning(unittest.TestCase):
    """ """
    def setUp(self):
        """ """
        # Define some example data to test on
        self.data = [
            {'id': 1, 'name': 'Alice', 'age': 28},
            {'id': 2, 'name': 'Bob', 'age': 35},
            {'id': 3, 'name': 'Charlie', 'age': None},
            {'id': 4, 'name': 'Dave', 'age': 42},
            {'id': 5, 'name': 'Eve', 'age': 29},
        ]

    def test_remove_duplicates(self):
        """ """
        # Test that the remove_duplicates function removes duplicates
        cleaned_data = cleaning.remove_duplicates(self.data)
        self.assertEqual(len(cleaned_data), len(
            set(tuple(d.items()) for d in self.data)))

    def test_fill_missing_values(self):
        """ """
        # Test that the fill_missing_values function fills missing values
        cleaned_data = cleaning.fill_missing_values(self.data)
        self.assertEqual([d for d in cleaned_data if d['age'] is None], [])


if __name__ == '__main__':
    unittest.main()

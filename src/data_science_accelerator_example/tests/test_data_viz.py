import unittest
from data_viz import plot_utils


class TestPlotUtils(unittest.TestCase):
    def setUp(self):
        # Define some example data to test on
        self.data = [
            {'id': 1, 'name': 'Alice', 'age': 28},
            {'id': 2, 'name': 'Bob', 'age': 35},
            {'id': 3, 'name': 'Charlie', 'age': 42},
            {'id': 4, 'name': 'Dave', 'age': 42},
            {'id': 5, 'name': 'Eve', 'age': 29},
        ]

    def test_barplot(self):
        # Test that the barplot function creates a plot
        plot = plot_utils.barplot(self.data, 'name', 'age')
        self.assertTrue(hasattr(plot, 'savefig'))


if __name__ == '__main__':
    unittest.main()

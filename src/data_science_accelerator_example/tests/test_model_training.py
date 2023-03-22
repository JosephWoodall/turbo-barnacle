import unittest
from model_training import model_selection


class TestModelSelection(unittest.TestCase):
    def setUp(self):
        # Define some example data to test on
        self.X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.y = [0, 1, 0]

    def test_train_test_split(self):
        # Test that the train_test_split function splits the data correctly
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            self.X, self.y)
        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_test), 1)


if __name__ == '__main__':
    unittest.main()

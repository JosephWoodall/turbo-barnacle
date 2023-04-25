import unittest
from data_pipeline.cleaning import Cleaning


class CleaningTest(unittest.TestCase):
    """
    The CleaningTest class tests the Cleaning class for execution of functions.
    """

    def __init__(self):
        self.Cleaning = Cleaning()
        print("------------------------------CLEANING_TEST_INITIALIZED")


if __name__ == '__main__':
    unittest.main()

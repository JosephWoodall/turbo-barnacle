import unittest
import numpy as np
from utils.config import load_config
from utils.logging import get_logger
from utils.metrics import calculate_rmse


class TestUtils(unittest.TestCase):
    """ """
    def test_load_config(self):
        """ """
        # test the load_config function
        config_path = 'path/to/config'
        config = load_config(config_path)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, dict)

    def test_get_logger(self):
        """ """
        # test the get_logger function
        logger_name = 'test_logger'
        logger = get_logger(logger_name)
        self.assertIsNotNone(logger)

    def test_calculate_rmse(self):
        """ """
        # test the calculate_rmse function
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        expected_output = 1.0
        self.assertEqual(calculate_rmse(y_true, y_pred), expected_output)

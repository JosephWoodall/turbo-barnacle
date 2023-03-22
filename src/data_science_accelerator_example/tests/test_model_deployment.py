import unittest
import numpy as np
import pandas as pd
from model_deployment.preprocessing import preprocess_data
from model_deployment.serving import load_model, make_prediction


class TestModelDeployment(unittest.TestCase):
    def test_preprocess_data(self):
        # test the preprocess_data function
        test_data = pd.DataFrame(
            {'feature1': [1, 2, 3], 'feature2': ['a', 'b', 'c']})
        expected_output = np.array([[1, 0, 1], [2, 1, 0], [3, 0, 0]])
        self.assertTrue(np.array_equal(
            preprocess_data(test_data), expected_output))

    def test_load_model(self):
        # test the load_model function
        model_path = 'path/to/model'
        model = load_model(model_path)
        self.assertIsNotNone(model)

    def test_make_prediction(self):
        # test the make_prediction function
        test_data = pd.DataFrame(
            {'feature1': [1, 2, 3], 'feature2': ['a', 'b', 'c']})
        model_path = 'path/to/model'
        model = load_model(model_path)
        prediction = make_prediction(test_data, model)
        self.assertIsInstance(prediction, np.ndarray)

import unittest
import pandas as pd
from great_expectations.dataset import PandasDataset

class TestValidation(unittest.TestCase):
    """ """
    def test_validation_operators(self):
        """ """
        data = pd.DataFrame(
            {
                "column1": [1, 2, 3],
                "column2": [1.1, 2.2, 3.3],
                "column3": ["a", "b", "c"],
            }
        )
        dataset = PandasDataset(data)

        # test unique column values
        self.assertEqual(dataset.expect_column_values_to_be_unique("column1"), None)
        self.assertEqual(dataset.expect_column_values_to_be_unique("column2"), None)

        # test column type
        self.assertEqual(dataset.expect_column_values_to_be_of_type("column1", "int"), None)
        self.assertEqual(dataset.expect_column_values_to_be_of_type("column2", "float"), None)

        # test column value match regex
        self.assertEqual(dataset.expect_column_values_to_match_regex("column3", "^[a-z]*$"), None)

    def test_validation_expectations(self):
        """ """
        data = pd.DataFrame(
            {
                "column1": [1, 2, 3],
                "column2": [1.1, 2.2, 3.3],
                "column3": ["a", "b", "c"],
            }
        )
        dataset = PandasDataset(data)

        # test unique column values
        self.assertEqual(dataset.validate_expectation(expectation_config={"expectation_type": "expect_column_values_to_be_unique", "column": "column1"}), None)
        self.assertEqual(dataset.validate_expectation(expectation_config={"expectation_type": "expect_column_values_to_be_unique", "column": "column2"}), None)

        # test column type
        self.assertEqual(dataset.validate_expectation(expectation_config={"expectation_type": "expect_column_values_to_be_of_type", "column": "column1", "type": "int"}), None)
        self.assertEqual(dataset.validate_expectation(expectation_config={"expectation_type": "expect_column_values_to_be_of_type", "column": "column2", "type": "float"}), None)

        # test column value match regex
        self.assertEqual(dataset.validate_expectation(expectation_config={"expectation_type": "expect_column_values_to_match_regex", "column": "column3", "regex": "^[a-z]*$"}), None)

if __name__ == '__main__':
    unittest.main()


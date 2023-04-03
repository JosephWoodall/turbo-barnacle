import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA


class Manova:
    """MANOVA: Use this test to compare the means of two or more continuous response variables across two or more groups."""

    def __init__(self, file_path, group_column, response_columns):
        """
        :param file_path: Path to the data file.
        :param group_column: Name of the column containing group labels.
        :param response_columns: List of names of the columns containing the response variables.
        """
        self.data = pd.read_csv(file_path)
        self.group_column = group_column
        self.response_columns = response_columns

    def run(self):
        """
        Fit and run the MANOVA model on the data.
        """
        response_data = self.data[self.response_columns]
        group_data = self.data[self.group_column]
        manova_model = MANOVA.from_formula(
            formula=" + ".join(self.response_columns) + f" ~ {self.group_column}", data=self.data)
        return manova_model

    def is_significant(self, alpha=0.05):
        """
        Test the significance of the MANOVA results.

        :param alpha: Significance level (Default value = 0.05)
        """
        manova_model = self.run()
        p_value = manova_model.mv_test(
        ).results['multivariate tests']['test statistic']['Wilks\' lambda']['p-value']
        if p_value < alpha:
            return True
        else:
            return False

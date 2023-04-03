import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA


class Manova:
    """MANOVA: Use this test to compare the means of two or more continuous response variables across two or more groups."""

    def __init__(self, file_path, explanatory_attribute, response_attributes):
        """_summary_

        Args:
            file_path (_type_): _description_
            explanatory_attribute (_type_): _description_
            response_attributes (_type_): _description_
        """
        self.data = pd.read_csv(file_path)
        self.explanatory_attribute = explanatory_attribute
        self.response_attributes = response_attributes

    def run(self):
        """Fit and run the MANOVA model on the data."""
        response_data = self.data[self.response_attributes]
        group_data = self.data[self.explanatory_attribute]
        manova_model = MANOVA.from_formula(
            formula=" + ".join(self.response_attributes) + f" ~ {self.explanatory_attribute}", data=self.data)
        return manova_model

    def is_significant(self, alpha=0.05):
        """Test the significance of the MANOVA results.

        :param alpha: Significance level (Default value = 0.05)

        """
        manova_model = self.run()
        p_value = manova_model.mv_test(
        ).results['multivariate tests']['test statistic']['Wilks\' lambda']['p-value']
        if p_value < alpha:
            return True
        else:
            return False

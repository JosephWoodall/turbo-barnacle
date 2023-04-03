import polars as pl
from scipy.stats import chi2_contingency


class ChiSquareTest:
    """Chi-squared test: Use this test to determine if there is a significant association between two categorical variables."""

    def __init__(self, file_path, explanatory_attribute, response_attribute):
        """_summary_

        Args:
            file_path (_type_): _description_
            explanatory_attribute (_type_): _description_
            response_attribute (_type_): _description_
        """
        self.data = pl.read_csv(file_path)
        self.explanatory_attribute = explanatory_attribute
        self.response_attribute = response_attribute
        self.contingency_table = pl.count(
            self.data, by=[explanatory_attribute, response_attribute]).pivot(explanatory_attribute, response_attribute)

    def run(self):
        """Fit and run the Chi-squared model on the data."""
        chi2, p_val, _, _ = chi2_contingency(self.contingency_table.to_numpy())
        return chi2, p_val

    def is_significant(self, alpha=0.05):
        """Test the significance of the Chi-square test results.

        :param alpha: Significance level (Default value = 0.05)

        """
        _, p_val = self.run()
        if p_val < alpha:
            return True
        else:
            return False

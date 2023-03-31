import polars as pl
from scipy.stats import chi2_contingency


class ChiSquareTest:
    """Chi-squared test: Use this test to determine if there is a significant association between two categorical variables.
    """

    def __init__(self, file_path, group_column, outcome_column):
        self.data = pl.read_csv(file_path)
        self.group_column = group_column
        self.outcome_column = outcome_column
        self.contingency_table = pl.count(
            self.data, by=[group_column, outcome_column]).pivot(group_column, outcome_column)

    def run(self):
        chi2, p_val, _, _ = chi2_contingency(self.contingency_table.to_numpy())
        return chi2, p_val

    def is_significant(self, alpha=0.05):
        _, p_val = self.run()
        if p_val < alpha:
            return True
        else:
            return False

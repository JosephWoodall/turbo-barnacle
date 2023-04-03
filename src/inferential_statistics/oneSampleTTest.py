import polars as pl
from scipy import stats


class OneSampleTTest:
    """One-sample t-test: Use this test to compare a sample mean to a known population mean."""

    def __init__(self, file_path, explanatory_attribute, response_attribute):
        self.data = pl.read_csv(file_path)
        self.explanatory_attribute = explanatory_attribute
        self.response_attribute = response_attribute
        self.group_1 = self.data.filter(
            pl.col(explanatory_attribute) == 1)[response_attribute]
        self.group_2 = self.data.filter(
            pl.col(explanatory_attribute) == 2)[response_attribute]

    def run(self):
        """ """
        t_stat, p_val = stats.ttest_ind(
            self.group_1.to_list(), self.group_2.to_list(), equal_var=False)
        return t_stat, p_val

    def is_significant(self, alpha=0.05):
        """

        :param alpha:  (Default value = 0.05)

        """
        _, p_val = self.run()
        if p_val < alpha:
            return True
        else:
            return False

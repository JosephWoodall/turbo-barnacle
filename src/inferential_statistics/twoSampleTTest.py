import polars as pl
from scipy.stats import ttest_ind


class TwoSampleTTest:
    """Two-sample t-test: Use this test to compare the means of two independent samples."""

    def __init__(self, file_path, group_column, value_column):
        self.data = pl.read_csv(file_path)
        self.group_column = group_column
        self.value_column = value_column
        self.group_1 = self.data.filter(
            pl.col(group_column) == 1)[value_column]
        self.group_2 = self.data.filter(
            pl.col(group_column) == 2)[value_column]

    def run(self):
        """ """
        _, p_val = ttest_ind(self.group_1.to_list(), self.group_2.to_list())
        return p_val

    def is_significant(self, alpha=0.05):
        """

        :param alpha:  (Default value = 0.05)

        """
        p_val = self.run()
        if p_val < alpha:
            return True
        else:
            return False

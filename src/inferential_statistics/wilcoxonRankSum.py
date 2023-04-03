import polars as pl
from scipy.stats import ranksums


class WilcoxonRankSumTest:
    """Wilcoxon rank-sum test: Use this test to compare the medians of two independent samples."""

    def __init__(self, file_path, group_column, value_column):
        """_summary_

        Args:
            file_path (_type_): _description_
            group_column (_type_): _description_
            value_column (_type_): _description_
        """
        self.data = pl.read_csv(file_path)
        self.group_column = group_column
        self.value_column = value_column
        self.group_1 = self.data.filter(
            pl.col(group_column) == 1)[value_column]
        self.group_2 = self.data.filter(
            pl.col(group_column) == 2)[value_column]

    def run(self):
        """Fit and run the Wilcoxon Rank Sum model on the data."""
        _, p_val = ranksums(self.group_1.to_list(), self.group_2.to_list())
        return p_val

    def is_significant(self, alpha=0.05):
        """Test the significance of the MANOVA results.

        :param alpha: Significance level (Default value = 0.05)

        """
        p_val = self.run()
        if p_val < alpha:
            return True
        else:
            return False

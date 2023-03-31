import polars as pl
from scipy.stats import ttest_rel


class PairedTTest:
    """Paired t-test: Use this test to compare the means of two dependent (paired) samples.
    """

    def __init__(self, file_path, pre_column, post_column):
        self.data = pl.read_csv(file_path)
        self.pre_column = pre_column
        self.post_column = post_column
        self.differences = self.data[post_column] - self.data[pre_column]

    def run(self):
        _, p_val = ttest_rel(
            self.data[self.pre_column], self.data[self.post_column])
        return p_val

    def is_significant(self, alpha=0.05):
        p_val = self.run()
        if p_val < alpha:
            return True
        else:
            return False

import polars as pl
from scipy.stats import kruskal


class KruskalWallisTest:
    """Kruskal-Wallis test: Use this test to compare the medians of three or more independent samples.
    """

    def __init__(self, file_path, group_column, value_column):
        self.data = pl.read_csv(file_path)
        self.group_column = group_column
        self.value_column = value_column

    def run(self):
        groups = [self.data[self.value_column][self.data[self.group_column] == g]
                  for g in self.data[self.group_column].unique()]
        _, p_val = kruskal(*groups)
        return p_val

    def is_significant(self, alpha=0.05):
        p_val = self.run()
        if p_val < alpha:
            return True
        else:
            return False

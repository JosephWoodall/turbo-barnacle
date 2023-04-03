import polars as pl
from scipy.stats import kruskal


class KruskalWallisTest:
    """Kruskal-Wallis test: Use this test to compare the medians of three or more independent samples."""

    def __init__(self, file_path, explanatory_attribute, response_attribute):
        self.data = pl.read_csv(file_path)
        self.explanatory_attribute = explanatory_attribute
        self.response_attribute = response_attribute

    def run(self):
        """ """
        groups = [self.data[self.response_attribute][self.data[self.explanatory_attribute] == g]
                  for g in self.data[self.explanatory_attribute].unique()]
        _, p_val = kruskal(*groups)
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

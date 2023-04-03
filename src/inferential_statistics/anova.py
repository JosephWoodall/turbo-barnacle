import polars as pl
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


class Anova:
    """ANOVA: Use this test to compare the means of three or more independent samples."""

    def __init__(self, file_path, explanatory_attribute, response_attribute):
        self.data = pl.read_csv(file_path)
        self.explanatory_attribute = explanatory_attribute
        self.response_attribute = response_attribute

    def run(self):
        """ """
        model = ols(f"{self.response_attribute} ~ C({self.explanatory_attribute})",
                    data=self.data.to_pandas()).fit()
        aov_table = anova_lm(model, typ=2)
        return aov_table

    def is_significant(self, alpha=0.05):
        """

        :param alpha:  (Default value = 0.05)

        """
        aov_table = self.run()
        if aov_table["PR(>F)"][0] < alpha:
            return True
        else:
            return False

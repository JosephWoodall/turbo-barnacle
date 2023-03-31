import polars as pl
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


class Anova:
    """ANOVA: Use this test to compare the means of three or more independent samples.
    """

    def __init__(self, file_path, group_column, value_column):
        self.data = pl.read_csv(file_path)
        self.group_column = group_column
        self.value_column = value_column

    def run(self):
        model = ols(f"{self.value_column} ~ C({self.group_column})",
                    data=self.data.to_pandas()).fit()
        aov_table = anova_lm(model, typ=2)
        return aov_table

    def is_significant(self, alpha=0.05):
        aov_table = self.run()
        if aov_table["PR(>F)"][0] < alpha:
            return True
        else:
            return False

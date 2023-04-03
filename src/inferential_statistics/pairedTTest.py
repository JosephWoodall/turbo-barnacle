import polars as pl
from scipy.stats import ttest_rel


class PairedTTest:
    """
    Paired t-test: Use this test to compare the means of two dependent (paired) samples.
    Here, the pre_column and post_column parameters are both attributes in the dataset, and their difference (self.differences)
    is used as the response variable. The pre_column variable represents the measurements or observations taken before an intervention
    or treatment, while the post_column varaible represents the measurements or observations taken after the intervention
    or treatment. Therefore, the pre_column parameter is the explanatory variable and the post_column is the response variable.
    """

    def __init__(self, file_path, pre_column, post_column):
        """_summary_

        Args:
            file_path (_type_): _description_
            pre_column (_type_): _description_
            post_column (_type_): _description_
        """
        self.data = pl.read_csv(file_path)
        self.pre_column = pre_column
        self.post_column = post_column
        self.differences = self.data[post_column] - self.data[pre_column]

    def run(self):
        """Fit and run the MANOVA model on the data."""
        _, p_val = ttest_rel(
            self.data[self.pre_column], self.data[self.post_column])
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

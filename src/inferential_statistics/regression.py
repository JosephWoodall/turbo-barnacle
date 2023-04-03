import polars as pl
import statsmodels.api as sm


class Regression:
    """ """

    def __init__(self, file_path, dependent_variable, independent_variables):
        """_summary_

        Args:
            file_path (_type_): _description_
            dependent_variable (_type_): _description_
            independent_variables (_type_): _description_
        """
        self.data = pl.read_csv(file_path)
        self.dependent_variable = dependent_variable
        self.independent_variables = independent_variables

    def run(self):
        """Fit and run the regression model on the data."""
        X = self.data[self.independent_variables]
        y = self.data[self.dependent_variable]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model

    def is_significant(self, alpha=0.05):
        """

        :param alpha: Default value = 0.05)

        """
        model = self.run()
        if model.pvalues[self.independent_variables].max() < alpha:
            return True
        else:
            return False

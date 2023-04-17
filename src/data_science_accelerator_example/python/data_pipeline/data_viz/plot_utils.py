import polars as pl
import matplotlib.pyplot as plt


class PlotUtils:
    """
     The PlotUtils class is a class that contains various functions to describe the specified data.
    """

    def __init__(self, data: pl.DataFrame):
        """
        __init__ Initializes the PlotUtils class

        Args:
            data (pl.DataFrame): _description_
        """
        self.data = data

    def plot_distribution(self, column: str) -> None:
        """
        plot_distribution plots the distribution of the specified data.

        Args:
            column (str): Name of the column to plot.
        """
        plt.hist(self.data[column], bins=20)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_correlation_matrix(self) -> None:
        """
        plot_correlation_matrix plots the correlation matrix of the specified data.
        """
        corr_matrix = self.data.corr()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(corr_matrix)
        plt.xticks(range(len(corr_matrix.columns)),
                   corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.show()

import polars as pl
import matplotlib.pyplot as plt


class PlotUtils:
    """ """
    def __init__(self, data):
        self.data = data

    def plot_distribution(self, column):
        """

        :param column: 

        """
        plt.hist(self.data[column], bins=20)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_correlation_matrix(self):
        """ """
        corr_matrix = self.data.corr()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(corr_matrix)
        plt.xticks(range(len(corr_matrix.columns)),
                   corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.show()

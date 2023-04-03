import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EDA:
    """ """
    def __init__(self, data):
        self.data = data

    def info(self):
        """ """
        # Return information about the data
        return self.data.info()

    def describe(self):
        """ """
        # Return summary statistics of the data
        return self.data.describe()

    def null_values(self):
        """ """
        # Return the percentage of null values in each column
        return round(self.data.isnull().mean()*100,2)

    def plot_hist(self, columns=None):
        """

        :param columns: Default value = None)

        """
        # Plot histograms of the specified columns
        if columns is None:
            self.data.hist()
        else:
            self.data[columns].hist()
        plt.show()
    
    def plot_box(self, columns=None):
        """

        :param columns: Default value = None)

        """
        # Plot box plots of the specified columns
        if columns is None:
            self.data.plot(kind='box')
        else:
            self.data[columns].plot(kind='box')
        plt.show()
        
    def plot_scatter(self, x, y):
        """

        :param x: param y:
        :param y: 

        """
        # Plot scatter plot of specified columns
        plt.scatter(x=self.data[x], y=self.data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
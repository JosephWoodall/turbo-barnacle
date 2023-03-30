class Validation:
    def __init__(self, data):
        self.data = data

    def check_missing_values(self):
        # count the number of missing values in each column
        num_missing = self.data.is_null().sum()
        # return columns with missing values
        return num_missing[num_missing > 0]

    def check_data_distribution(self):
        # code to check the distribution of data
        pass

    def check_correlation(self):
        # calculate the correlation matrix
        corr_matrix = self.data.corr()
        # return the correlation matrix
        return corr_matrix

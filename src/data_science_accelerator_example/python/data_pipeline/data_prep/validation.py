class Validation:
    """
     The Validation class will perform various validation functions on the specified data source. Please note, the functions may need to be customized depending on the use case.
    """

    def __init__(self, data):
        """
        __init__ Initializes the Validation class. 

        Args:
            data (_type_): _description_
        """
        self.data = data

    def check_missing_values(self):
        """
        check_missing_values checks for any missing values of the specified data.

        Returns:
            _type_: _description_
        """
        # count the number of missing values in each column
        num_missing = self.data.is_null().sum()
        # return columns with missing values
        return num_missing[num_missing > 0]

    def check_data_distribution(self):
        """ """
        # code to check the distribution of data
        pass

    def check_correlation(self):
        """ """
        # calculate the correlation matrix
        corr_matrix = self.data.corr()
        # return the correlation matrix
        return corr_matrix

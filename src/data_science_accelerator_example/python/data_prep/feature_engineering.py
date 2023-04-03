import polars as pl


class FeatureEngineering:
    """
     The FeatureEngineering class creates new features in the data for downstream use in predictive modeling or inferential statistics.
    """

    def __init__(self, data: pl.DataFrame):
        """
        __init__ Initializes the FeatureEngineering class.

        Args:
            data (pl.DataFrame): the object representing the data used in the functions of this class.
        """
        self.data = data

    def add_new_feature(self):
        """
        add_new_feature creates new features in the data. 
        """
        # code to add a new feature to the data
        pass

    def extract_datetime_features(self):
        """
        extract_datetime_features extracts the datetime features from the data.
        """
        # code to extract datetime features from a column
        pass

import polars as pl
from sklearn.preprocessing import OneHotEncoder


class Transformation:
    """
     The Transformation class will perform various transformations onto the specified dataset.
    """

    def __init__(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        __init__ Initializes the Transformation class.

        Args:
            data (pl.DataFrame): the class level data source to be used by the functions of the Transformation class.

        Returns:
            pl.DataFrame: the class level data source to be used by the functions of the Transformation class.
        """
        self.data = pl.DataFrame(data)

    def scale_features(self) -> pl.DataFrame:
        """
        scale_features scales the features of the continuous attributes in the specified data source.


        Returns:
            pl.DataFrame: data source with scaled continuous attribute levels.
        """
        # code to scale features
        pass

    def normalize_features(self) -> pl.DataFrame:
        """
        normalize_features normalizes the continuous attributes in the specified data source.

        Returns:
            pl.DataFrame: data source with normalizes continuous attribute levels.
        """
        # code to normalize features
        pass

    def one_hot_encode(self, column: pl.Series[str]):
        """

        :param column: 

        """
        self.data = self.data.one_hot(column)

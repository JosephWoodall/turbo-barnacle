import polars as pl


class Transformation:
    def __init__(self, data):
        self.data = pl.DataFrame(data)

    def scale_features(self):
        # code to scale features
        pass

    def normalize_features(self):
        # code to normalize features
        pass

    def one_hot_encode(self, column):
        self.data = self.data.one_hot(column)

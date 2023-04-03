import polars as pl
from sklearn.ensemble import RandomForestRegressor


class Cleaning:
    """ """

    def __init__(self, data):
        self.data = pl.DataFrame(data)

    def remove_duplicates(self):
        """ """
        self.data.drop_duplicates(inplace=True)

    def remove_null_values(self):
        """ """
        self.data.dropna(inplace=True)

    def fill_null_values(self):
        """Fill null values for both categorical and continuous attributes."""
        # Separate the data into non-null and null values
        non_null_data = self.data.dropna()
        null_data = self.data.filter(lambda df: df.is_null().any(), None)

        # Fill null values for continuous attributes
        continuous_cols = non_null_data.select_dtypes(pl.datatypes.Float64)
        if not continuous_cols.is_empty():
            # Train the Random Forest model on the non-null data
            X_train = non_null_data[continuous_cols.names()]
            y_train = non_null_data['target_variable']
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Use the trained model to predict the null values
            X_test = null_data[continuous_cols.names()]
            y_pred = model.predict(X_test)

            # Replace the null values with the predicted values
            null_data[continuous_cols.names()] = pl.DataFrame(
                y_pred, nullable=True)

        # Fill null values for categorical attributes
        categorical_cols = non_null_data.select_dtypes(
            pl.datatypes.Categorical)
        if not categorical_cols.is_empty():
            # Fill null values with the most frequent category
            for col in categorical_cols.names():
                mode = non_null_data[col].mode()[0]
                null_data[col] = null_data[col].fillna(mode)

        # Combine the non-null and filled null data
        self.data = pl.concat([non_null_data, null_data])

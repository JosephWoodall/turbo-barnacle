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
        """ """
        # Separate the data into non-null and null values
        non_null_data = self.data.dropna()
        null_data = self.data.filter(lambda df: df.is_null().any(), None)

        # Train the Random Forest model on the non-null data
        X_train = non_null_data.drop('target_variable')
        y_train = non_null_data['target_variable']
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Use the trained model to predict the null values
        X_test = null_data.drop('target_variable')
        y_pred = model.predict(X_test)

        # Replace the null values with the predicted values
        self.data['target_variable'] = self.data['target_variable'].if_else(
            self.data['target_variable'].is_null(),
            pl.Series(y_pred, nullable=True),
            self.data['target_variable']
        )

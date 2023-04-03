from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

class RandomForestModel:
    """ """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = None

    def train_model(self, n_iter=10, cv=5):
        """

        :param n_iter:  (Default value = 10)
        :param cv:  (Default value = 5)

        """
        # Define the parameter grid for random search
        param_grid = {
            "n_estimators": np.arange(10, 101, 10),
            "max_depth": np.arange(2, 11, 1),
            "min_samples_leaf": np.arange(1, 6, 1),
            "min_samples_split": np.arange(2, 11, 1),
        }

        # Initialize the classifier and random search
        rf = RandomForestClassifier()
        self.model = RandomizedSearchCV(rf, param_grid, n_iter=n_iter, cv=cv)

        # Fit the classifier to the data
        self.model.fit(self.data, self.target)

    def predict(self, data):
        """

        :param data: 

        """
        # Use the trained model to make predictions
        return self.model.predict(data)

    def get_best_params(self):
        """ """
        # return best parameters found by the RandomizedSearchCV
        return self.model.best_params_

    def get_best_estimator(self):
        """ """
        # return the best estimator
        return self.model.best_estimator_

    def evaluate_model(self, x_test, y_test):
        """

        :param x_test: 
        :param y_test: 

        """
        # Make predictions on the test set
        y_pred = self.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print("Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%".format(acc*100, precision*100, recall*100))
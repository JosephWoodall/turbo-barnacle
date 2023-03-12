from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

"""
The ModelTrainer class with some basic functionality that can be used to train a set of models and evaluate them. 
The class initializes with the data and target variables and creates a dictionary of models with default hyperparameters.

The train_models method uses GridSearchCV to perform hyperparameter tuning on each of the models and stores the best estimator 
for each model in a dictionary best_models.

The evaluate_models method takes in test data and target and uses the best_models to predict on test data and prints accuracy, 
precision and recall for each of the models.

The predict method takes in new data and returns the predictions of all the models stored in best_models dictionary.

In the last part of script, we are instantiating the class with training data and target, training models with 
the provided parameters, evaluating the models on test data and predicting new data using the models.

It's important to note that the script uses scikit-learn library for training machine learning models, this script 
is just an example and you can modify it to use any machine learning library or any other model you have in mind.
"""

class ModelTrainer:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(),
            "KNeighbors": KNeighborsClassifier()
        }
        self.best_models = {}

    def train_models(self, parameters):
        for model_name, model in self.models.items():
            gs = GridSearchCV(model, parameters[model_name], cv=5)
            gs.fit(self.data, self.target)
            self.best_models[model_name] = gs.best_estimator_
            
    def evaluate_models(self, x_test, y_test):
        for model_name, model in self.best_models.items():
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            print(f"{model_name} Model:")
            print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")
            
    def predict(self, x_test):
        predictions = {}
        for model_name, model in self.best_models.items():
            predictions[model_name] = model.predict(x_test)
        return predictions

parameters = {
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20],
        "min_samples_leaf": [1, 2, 4]
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"]
    },
    "KNeighbors": {
        "n_neighbors": [5, 10, 15],
        "weights": ["uniform", "distance"]
    }
}

train_data = '' # insert your data here
train_target = ''  # insert your target variable here

trainer = ModelTrainer

(train_data, train_target)
trainer.train_models(parameters)

x_test = ''  # insert your test data here
y_test = ''  # insert your test target here

trainer.evaluate_models(x_test, y_test)

new_data = ''  # insert new data on which you want to make predictions
predictions = trainer.predict(new_data)
print(predictions)
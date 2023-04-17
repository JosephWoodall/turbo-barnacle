import time
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data_science_accelerator_example.python.model_training.model_selection import ModelSelection


class ModelMonitor:
    def __init__(self, model_url, threshold_accuracy=0.8, threshold_precision=0.8, threshold_recall=0.8):
        self.model_url = model_url
        self.threshold_accuracy = threshold_accuracy
        self.threshold_precision = threshold_precision
        self.threshold_recall = threshold_recall
        self.model = ModelSelection()  # instantiate your model class here

    def predict(self, input_data):
        # sends prediction request to the deployed model
        response = requests.post(self.model_url, json=input_data)
        return response.json()

    def monitor(self, input_data, expected_output):
        # evaluate model on new data
        predicted_output = self.predict(input_data)
        accuracy = accuracy_score(expected_output, predicted_output)
        precision = precision_score(expected_output, predicted_output)
        recall = recall_score(expected_output, predicted_output)

        # check if model meets minimum performance threshold
        if accuracy < self.threshold_accuracy or precision < self.threshold_precision or recall < self.threshold_recall:
            print("Model performance has decreased below threshold. Retraining model...")
            self.retrain_model(input_data, expected_output)

    def retrain_model(self, input_data, expected_output):
        # retrain model on new data
        self.model.fit(input_data, expected_output)
        self.model.deploy()  # deploy the new version of the model

    def run(self, input_data, expected_output):
        # run the monitoring loop
        while True:
            self.monitor(input_data, expected_output)


'''
# Example usage
model_url = "http://my-deployed-model.com/predict"
threshold_accuracy = 0.75
threshold_precision=0.75
threshold_recall=0.75
monitor = ModelMonitor(model_url = model_url, 
    threshold_accuracy = threshold_accuracy, 
    threshold_precision = threshold_precision,
    threshold_recall = threshold_recall)

input_data = ""
expected_output = ""

monitor.run(input_data, expected_output)
'''

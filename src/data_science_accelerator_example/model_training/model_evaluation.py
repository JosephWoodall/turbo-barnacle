import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ModelEvaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred)

    def f1(self):
        return f1_score(self.y_true, self.y_pred)

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def metrics_summary(self):
        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'F1': self.f1()
        }

        return pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])

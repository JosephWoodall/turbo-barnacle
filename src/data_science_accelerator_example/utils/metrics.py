import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix


def calculate_metrics(y_true, y_pred):
    """Calculates a dictionary of regression metrics given true and predicted values.

    :param y_true: 
    :param y_pred: 

    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {'mse': mse, 'mae': mae, 'r2': r2}


def calculate_classification_metrics(y_true, y_pred):
    """Calculates a dictionary of classification metrics given true and predicted values.

    :param y_true: 
    :param y_pred: 

    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}

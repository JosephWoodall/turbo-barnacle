'''
includes best model selection, model performance metrics (accuracy, precision, recall, f1, etc...)
'''
import inspect


class ModelEvaluation:
    def __init__(self):
        pass

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self


model_evaluation = ModelEvaluation()
model_evaluation.call_all_methods()

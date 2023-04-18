'''
includes feature engineering, hyperparameter tuning
'''
import h2o
from h2o.automl import H2OAutoML
import inspect


class ModelEngineering:

    def __init__(self, explanatory_var=[], response_var=[]):
        self.explanatory_var = explanatory_var
        self.response_var = response_var

    def _feature_engineering(self):
        """
        _feature_engineering generates features used for model evaluation
        """
        pass

    def _model_selection(self):
        """
        _model_selection tests a number of models based on training and testing metrics
        """
        h2o.init()
        aml = H2OAutoML(max_models=20, seed=1)
        aml.train(x=self.explanatory_var,
                  y=self.response_var, training_frame="")  # training_frame needs to be the train object from the DataVersioning class
        return aml.leader

    def _hyperparameter_tuning(self):
        """
        _hyperparameter_tuning optimizes hyperparameters of selected model
        """
        pass

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self


model_engineering = ModelEngineering()
model_engineering.call_all_methods()

'''
includes logs and checks for model decay, outputs feedback on new data from model performance
'''
import inspect


class MonitoringAndLogging:
    def __init__(self):
        pass

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self


monitoring_and_logging = MonitoringAndLogging()
monitoring_and_logging.call_all_methods()

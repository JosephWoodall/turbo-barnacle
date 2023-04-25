'''
'''

import inspect


class MonitoringAndLoggingTest:
    """
    The MonitoringAndLoggingTestTest class tests the MonitoringAndLogging class for execution of functions.
    """

    def __init__(self):
        print("------------------------------MONITORING_AND_LOGGING_TEST_INITIALIZED")

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self

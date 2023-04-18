'''
retrieves the source data for the pipeline and returns the data in one conoslidated location
'''
import inspect


class SourceDataRetrieval:
    def __init__(self):
        pass

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self


source_data_retireval = SourceDataRetrieval()
source_data_retireval.call_all_methods()

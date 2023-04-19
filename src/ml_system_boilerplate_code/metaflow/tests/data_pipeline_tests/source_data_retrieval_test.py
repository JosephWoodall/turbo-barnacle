import inspect


class SourceDataRetrievalTest:
    """
    The SourceDataRetrievalTest class tests the SourceDataRetrieval class for execution of functions.

    """

    def __init__(self):
        pass

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self


source_data_retrieval_test = SourceDataRetrievalTest()
source_data_retrieval_test.call_all_methods()

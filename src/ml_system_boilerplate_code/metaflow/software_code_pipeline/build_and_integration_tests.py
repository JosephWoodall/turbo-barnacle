'''
includes any and all build tests, integration tests, etc... outputs a green-light served model for production
'''
import inspect


class BuildAndIntegrationTests:

    def __init__(self):
        pass

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self


build_and_integration_tests = BuildAndIntegrationTests()
build_and_integration_tests.call_all_methods()

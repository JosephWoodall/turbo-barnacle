'''
exposes the development environment model to the production environment
'''
import inspect


class DeploymentDevelopmentToProduction:

    def __init__(self):
        print("-----DEPLOYMENT DEVELOPMENT TO PRODUCTION INITIALIZED-----")

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self

'''
exposes the development environment model to the production environment
'''
import inspect


class DeploymentDevelopmentToProduction:

    def __init__(self):
        print("-----DEPLOYMENT DEVELOPMENT TO PRODUCTION INITIALIZED-----")


if __name__ == '__main__':
    DeploymentDevelopmentToProduction()

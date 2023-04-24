from metaflow import FlowSpec, Run


from tests.data_pipeline_tests.cleaning_test import CleaningTest
from tests.data_pipeline_tests.data_versioning_test import DataVersioningTest
from tests.data_pipeline_tests.exploration_and_validation_test import ExplorationAndValidationTest
from tests.data_pipeline_tests.source_data_retrieval_test import SourceDataRetrievalTest

from tests.machine_learning_pipeline_tests.model_engineering_test import ModelEngineeringTest
from tests.machine_learning_pipeline_tests.model_evaluation_test import ModelEvaluationTest
from tests.machine_learning_pipeline_tests.model_packaging_test import ModelPackagingTest
from tests.machine_learning_pipeline_tests.model_test import ModelTest

from tests.software_code_pipeline_tests.build_and_integration_test import BuildAndIntegrationTest
from tests.software_code_pipeline_tests.deployment_dev_to_production_test import DeploymentDevelopmentToProductionTest
from tests.software_code_pipeline_tests.monitoring_and_logging_test import MonitoringAndLoggingTest

import inspect


class MainTest:

    def __init__(self):
        pass

    def call_all_methods(self):
        # gets a list of all the functions defined in the class
        functions = inspect.getmembers(self, predicate=inspect.ismethod)
        # iterates through the list of functions and calls each one on self
        for func in functions:
            func()  # calls the function on self

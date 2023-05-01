from data_pipeline.cleaning import Cleaning
from data_pipeline.data_versioning import DataVersioning
from data_pipeline.exploration_and_validation import ExplorationAndValidation
from data_pipeline.source_data_retrieval import SourceDataRetrieval

from machine_learning_pipeline.model import Model
from machine_learning_pipeline.model_engineering import ModelEngineering
from machine_learning_pipeline.model_evaluation import ModelEvaluation
from machine_learning_pipeline.model_packaging import ModelPackaging

from software_code_pipeline.build_and_integration_tests import BuildAndIntegrationTests
from software_code_pipeline.deployment_dev_to_production import DeploymentDevelopmentToProduction
from software_code_pipeline.monitoring_and_logging import MonitoringAndLogging

from tests.main_test import MainTest


class Main():
    pass


if __name__ == "__main__":
    main_flow = Main()

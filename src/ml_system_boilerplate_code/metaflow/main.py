'''
defines the pipeline components to be run in the specified order
'''

from data_pipeline.cleaning import Cleaning
from data_pipeline.data_versioning import DataVersioning
from data_pipeline.exploration_and_validation import ExplorationAndValidation
from data_pipeline.source_data_retrieval import SourceDataRetrieval

from machine_learning_pipeline.model import Model
from machine_learning_pipeline.model_engineering import ModelEngineering
from machine_learning_pipeline.model_evaluation import ModelEvaluation
from machine_learning_pipeline.model_packaging import ModelPackaging

from software_code_pipeline.build_and_integration_tests import BuildAndIntegrationTests
from software_code_pipeline.deployment_dev_to_production import DeploymentDevToProduction
from software_code_pipeline.monitoring_and_logging import MonitoringAndLogging

from metaflow import FlowSpec, step


class Main(FlowSpec):
    """
    Main runs the machine learning microservice according to the specified workflow.

    This class will define the transition outlined in diagram of the README.md

    Workflow: 
    PROGRAM START
    |
    V
    DATA PIPELINE ENTRY: 
    |
    V
        - data_pipeline.source_data_retrieval - SourceDataRetrieval
    |
    V
        - data_pipeline.exploration_and_validation - ExplorationAndValidation
    |
    V
    (***)- data_pipeline.cleaning - Cleaning (reiterate target is: data_pipeline.exploration_and_validation - ExplorationAndValidation)
    |
    V
        - data_pipeline.data_versioning - DataVersioning 
    |
    V
    END OF DATA PIPELINE PHASE
    |
    V
    MACHINE LEARNING PIPELINE ENTRY:
    |
    V
        - machine_learning_pipeline.model_engineering - ModelEngineering (reiterate target is: data_pipeline.source_data_retrieval - SourceDataRetrieval)
    |
    V
        - machine_learning_pipeline.model_evaluation - ModelEvaluation (reiterate target is: machine_learning_pipeline.model_engineering - ModelEngineering)
    |
    V
        - machine_learning_pipeline.model_packaging - ModelPackaging
    |
    V
        - machine_learning_pipeline.model - Model
    |
    v
    END OF MACHINE LEARNING PIPELINE PHASE
    |
    v
    SOFTWARE CODE PIPELINE ENTRY:
    |
    V
        - software_code_pipeline.build_and_integration_tests - BuildAndIntegrationTests
    |
    V
        - software_code_pipeline.deployment_dev_to_production - DeploymentDevToProduction
    |
    V
        - software_code_pipeline.monitoring_and_logging - MonitoringAndLogging
    |
    V
    END OF SOFTWARE CODE PIPELINE PHASE
    |
    V
    RE-ENTRY TO DATA PIPELINE ON SET CADENCE (TRIGGERED BY MODEL DECAY TRIGGER)
    |
    V
        - data_pipeline.cleaning - Cleaning
    |
    V
        - REPEAT PIPELINE N-TIMES STARTING AT *** STEP

    """
    pass


if __name__ == "__main__":
    Main()

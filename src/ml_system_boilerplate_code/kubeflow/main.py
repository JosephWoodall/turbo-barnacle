from data_pipeline.data_pipeline_handler import DataPipeline
from machine_learning_pipeline.machine_learing_pipeline_handler import MachineLearningPipeline
from software_code_pipeline.software_code_pipeline_handler import SoftwareCodePipeline
'''





from tests.main_test import MainTest
'''
import kfp
from kfp import dsl, compiler
import kfp.components as comp


class MainPipeline():
    """
    MainPipeline runs the machine learning microservice according to the specified workflow, as such, this class defines the pipeline components to be run in the specified order.

    This class will define the transition outlined in diagram of the README.md.

    Legend: 
        - reiterate targets triggered by their home step and traverse the pipeline, recursively, if their criteria is not satisfied, specified as below: 
            - 1 = send back to reiterate target
            - 0 = do not send back to reiterate target
    Pre-Check Tests will send back to specific component steps if/when a component does not pass its check.

    Workflow: 
    PROGRAM START
    |
    V
        - PRE-CHECK TEST: Run tests.main_test.MainTest test class if all steps have required inputs/functionality to successfully execute.
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
        - machine_learning_pipeline.model_evaluation - ModelEvaluation (reiterate target is: machine_learning_pipeline.model_engineering - ModelEngineering) {this is a Continual Learning implementation}
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

    def __init__(self):
        self.DataPipeline = DataPipeline
        self.MachineLearningPipeline = MachineLearningPipeline
        self.SoftwareCodePipeline = SoftwareCodePipeline

    @dsl.pipeline(name="ml_system_boilerplate_code_pipeline",
                  description="templatized pipeline ftw",
                  display_name="ml_system_boilerplate_code_pipeline")
    def run_pipeline(self):
        with dsl.Condition(self.DataPipeline() == 1):
            print("DATA PIPELINE RAN SUCCESSFULLY")
            with dsl.Condition(self.MachineLearningPipeline() == 1):
                print("MACHINE LEARNING PIPELINE RAN SUCCESSFULLY")
                with dsl.Condition(self.SoftwareCodePipeline() == 1):
                    print("SOFTWARE CODE PIPELINE RAN SUCCESSFULLY")


if __name__ == "__main__":
    main = MainPipeline()

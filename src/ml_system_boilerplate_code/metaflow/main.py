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

from metaflow import FlowSpec, step, project, schedule, card


ML_SYSTEM_SERVICE_NAME = "<INSERT_SERVICE_NAME_HERE>"
# import your custom cron schedule here, store cron schedule in this project
CRON_SCHEDULE = ''


@schedule(CRON_SCHEDULE)
@project(name=ML_SYSTEM_SERVICE_NAME)
class Main(FlowSpec):
    """
    Main runs the machine learning microservice according to the specified workflow, as such, this class defines the pipeline components to be run in the specified order.

    This class will define the transition outlined in diagram of the README.md.

    Legend: 
        - reiterate targets triggered by their home step and traverse the pipeline, recursively, if their criteria is not satisfied, specified as below: 
            - 1 = send back to reiterate target
            - 0 = do not send back to reiterate target

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
        self.MainTest = MainTest()

        self.Cleaning = Cleaning()
        self.DataVersioning = DataVersioning()
        self.ExplorationAndValidation = ExplorationAndValidation()
        self.SourceDataRetrieval = SourceDataRetrieval()

        self.Model = Model()
        self.ModelEngineering = ModelEngineering()
        self.ModelEvaluation = ModelEvaluation()
        self.ModelPackaging = ModelPackaging()

        self.BuildAndIntegrationTests = BuildAndIntegrationTests()
        self.DeploymentDevToProduction = DeploymentDevelopmentToProduction()
        self.MonitoringAndLogging = MonitoringAndLogging()

    '''PRE-CHECK TEST'''
    @step
    def main_test(self):
        """
        main_test executes the pre-check tests to test if the functions below have their required inputs/functionality 

        ####THIS STEP NEEDS A REWORK, JUST WROTE THIS IN FOR PLACEHOLDER PURPOSES####
        """
        try:
            if self.MainTest.call_all_methods():
                self.next(self.source_data_retrieval)
        except ValueError as ve:
            print("At least one step has failed its pre-check. Please check Metaflow checks for step(s) that failed to pass pre-check!")
            exit()

    '''DATA PIPELINE'''
    @step
    def source_data_retrieval(self):
        """
        source_data_retrieval executes the SourceDataRetrieval class
        """
        self.SourceDataRetrieval.call_all_methods()
        self.next(self.exploration_and_validation)

    @step
    def exploration_and_validation(self):
        """
        exploration_and_validation executes the ExplorationAndValidation class
        """
        self.ExplorationAndValidation.call_all_methods()
        self.next(self.cleaning)

    @step
    def cleaning(self):
        """
        cleaning executes the Cleaning class.
        """
        send_back_to_exploration_and_validation_criteria = ''
        if send_back_to_exploration_and_validation_criteria != 0:
            self.next(self.exploration_and_validation)
        else:
            self.Cleaning.call_all_methods()
            self.next(self.data_versioning)

    @step
    def data_versioning(self):
        """
        data_versioning executes the DataVersioning class
        """
        self.DataVersioning.call_all_methods()
        self.next(self.model_engineering)

    '''MACHINE LEARNING PIPELINE'''
    @step
    def model_engineering(self):
        """
        model_engineering executes the ModelEngineering class
        """
        send_back_to_source_data_retrieval_criteria = ''
        if send_back_to_source_data_retrieval_criteria != 0:
            self.next(self.source_data_retrieval)
        else:
            self.ModelEngineering.call_all_methods()
            self.next(self.model_evaluation)

    @step
    def model_evaluation(self):
        """
        model_evaluation executes the ModelEvaluation class
        """
        send_back_to_model_engineering_criteria = ''
        if send_back_to_model_engineering_criteria != 0:
            self.next(self.model_engineering)
        else:
            self.ModelEvaluation.call_all_methods()
            self.next(self.model_packaging)

    @step
    def model_packaging(self):
        """
        model_packaging executes the ModelPackaging class
        """
        self.ModelPackaging.call_all_methods()
        self.next(self.model)

    @step
    def model(self):
        """
        model executes the Model class

        includes check if the data (from DataVersioning) and model (from Model) have the same pretense version number.
        """
        data_versioning_pretense = ''
        model_versioning_pretense = ''
        try:
            data_versioning_pretense != model_versioning_pretense
        except ValueError as ve:
            pass  # print the value error here or return it to the debug board?
        self.Model.call_all_methods()
        self.next(self.build_and_integration_tests)

    '''SOFTWARE CODE PIPELINE'''
    @step
    def build_and_integration_tests(self):
        """
        build_and_integration_tests executes the BuildAndIntegrationTests class
        """
        self.BuildAndIntegrationTests.call_all_methods()
        self.next(self.deployment_dev_to_production)

    @step
    def deployment_development_to_production(self):
        """
        deployment_dev_to_production executes the DeploymentDevToProduction class
        """
        self.DeploymentDevToProduction.call_all_methods()
        self.next(self.monitoring_and_logging)

    @step
    @card
    def monitoring_and_logging(self):
        """
        monitoring_and_logging executes the MonitoringAndLogging class. Model decay trigger is included here.
        """
        while True:
            self.MonitoringAndLogging.call_all_methods()
            send_back_to_cleaning_criteria_model_decay_trigger = ''
            if send_back_to_cleaning_criteria_model_decay_trigger != 0:
                continue
            self.next(self.cleaning)


if __name__ == "__main__":
    Main()

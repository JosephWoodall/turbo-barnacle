from software_code_pipeline.build_and_integration_tests import BuildAndIntegrationTests
from software_code_pipeline.deployment_dev_to_production import DeploymentDevelopmentToProduction
from software_code_pipeline.monitoring_and_logging import MonitoringAndLogging

import kfp
from kfp import dsl, compiler
import kfp.components as comp


class SoftwareCodePipeline():

    def __init__(self):
        self.BuildAndIntegrationTests = BuildAndIntegrationTests
        self.DeploymentDevelopmentToProduction = DeploymentDevelopmentToProduction
        self.MonitoringAndLogging = MonitoringAndLogging

    @dsl.component()
    def _build_and_integration_tests(self):
        self.BuildAndIntegrationTests()

    @dsl.component()
    def _deployment_development_to_production(self):
        self.DeploymentDevelopmentToProduction()

    @dsl.component()
    def _monitoring_and_logging(self):
        self.MonitoringAndLogging()

    def run_pipeline(self):
        print("RUNNING SOFTWARE CODE PIPELINE")
        pass
        # check pre-test checks criteria here
        # with dsl.Condition():
        #   return 1
        # check if productionized object output is present and has the three module tags associated with it

import kfp
from kfp import dsl, compiler
import kfp.components as comp


class SoftwareCodePipeline(dsl.Pipeline):

    def __init__(self):
        from build_and_integration_tests import BuildAndIntegrationTests
        from deployment_dev_to_production import DeploymentDevelopmentToProduction
        from monitoring_and_logging import MonitoringAndLogging

        self.BuildAndIntegrationTests = BuildAndIntegrationTests
        self.DeploymentDevelopmentToProduction = DeploymentDevelopmentToProduction
        self.MonitoringAndLogging = MonitoringAndLogging
        self.build_deployment_monitoring_object = None

    @dsl.component
    def __build_and_integration_tests(self):
        self.BuildAndIntegrationTests()

    @dsl.component
    def __deployment_development_to_production(self):
        self.DeploymentDevelopmentToProduction()

    @dsl.component
    def __monitoring_and_logging(self):
        self.MonitoringAndLogging()

    @dsl.component
    def _build_deployment_monitoring_object(self):
        self.build_deployment_monitoring_object = {
            "data_object": "",
            "test_pass": 1
        }
        print(self.build_deployment_monitoring_object.values())
        return self.build_deployment_monitoring_object.values()

    @dsl.pipeline
    def run_pipeline(self):
        print("RUNNING SOFTWARE CODE PIPELINE")
        return SoftwareCodePipeline._build_deployment_monitoring_object().outputs

        # check pre-test checks criteria here
        # with dsl.Condition():
        #   return 1
        # check if productionized object output is present and has the three module tags associated with it


if __name__ == "__main__":
    import os
    os.chdir(r'./src/ml_system_boilerplate_code/kubeflow/software_code_pipeline/')
    compiler.Compiler().compile(SoftwareCodePipeline.run_pipeline,
                                package_path='software_code_pipeline.yaml')

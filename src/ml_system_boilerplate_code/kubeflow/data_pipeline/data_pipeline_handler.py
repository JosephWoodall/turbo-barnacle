import kfp
from kfp import dsl, compiler
import kfp.components as comp


class DataPipeline(dsl.ContainerOp):

    def __init__(self):
        from cleaning import Cleaning
        from data_versioning import DataVersioning
        from exploration_and_validation import ExplorationAndValidation
        from source_data_retrieval import SourceDataRetrieval

        self.SourceDataRetrieval = SourceDataRetrieval
        self.ExplorationAndValidation = ExplorationAndValidation
        self.Cleaning = Cleaning
        self.DataVersioning = DataVersioning
        self.data_object = None

    @dsl.component
    def __source_data_retrieval(self):
        return self.SourceDataRetrieval()._fake_data_generator(2, 2)

    @dsl.component
    def __exploration_and_validation(self):
        return self.ExplorationAndValidation()

    @dsl.component
    def __cleaning(self):
        return self.Cleaning()

    @dsl.component
    def __data_versioning(self):
        return self.DataVersioning()

    @dsl.component
    def _data_object(self) -> dict:

        # populated by the above components, using defined value for testing
        self.data_object = {
            "data_object": "",
            "test_pass": 1
        }
        print(self.data_object.values()
              )
        return self.data_object.values()

    @dsl.pipeline
    def run_pipeline(self):
        print("--RUNNING DATA PIPELINE--")
        return DataPipeline()._data_object().outputs
        # check pre-test checks criteria here
        # with dsl.Condition(self._data_object().output['test_pass'] == 1):
        #    return self._data_object().output['data_object'].values()
        # check data object output is present


if __name__ == "__main__":
    import os
    os.chdir(r'./src/ml_system_boilerplate_code/kubeflow/data_pipeline/')
    print(os.getcwd())
    compiler.Compiler().compile(DataPipeline().run_pipeline(),
                                package_path='machine_learning_pipeline.yaml')

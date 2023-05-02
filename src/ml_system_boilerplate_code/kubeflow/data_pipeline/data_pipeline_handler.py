from data_pipeline.cleaning import Cleaning
from data_pipeline.data_versioning import DataVersioning
from data_pipeline.exploration_and_validation import ExplorationAndValidation
from data_pipeline.source_data_retrieval import SourceDataRetrieval

import kfp
from kfp import dsl, compiler
import kfp.components as comp


class DataPipeline:

    def __init__(self):
        self.SourceDataRetrieval = SourceDataRetrieval
        self.ExplorationAndValidation = ExplorationAndValidation
        self.Cleaning = Cleaning
        self.DataVersioning = DataVersioning

    @dsl.component
    def _source_data_retrieval(self):
        return self.SourceDataRetrieval._fake_data_generator(2, 2)

    @dsl.component
    def _exploration_and_validation(self):
        return self.ExplorationAndValidation()

    @dsl.component
    def _cleaning(self):
        return self.Cleaning._cleaning_process_one(2)

    @dsl.component
    def _data_versioning(self):
        return self.DataVersioning()

    @dsl.component
    def _data_object(self) -> dict:
        # populated by the above components, using defined value for testing
        data_object = {
            "data_object": "",
            "test_pass": 1
        }
        return data_object

    @dsl.pipeline(name="DATA_PIPELINE_ml_system_boilerplate_code_pipeline",
                  description="templatized pipeline ftw",
                  )
    def run_pipeline(self):
        print("RUNNING DATA PIPELINE")
        # check pre-test checks criteria here
        with dsl.Condition(self._data_object().output['test_pass'] == 1):
            return self._data_object().output['data_object'].values()
        # check data object output is present

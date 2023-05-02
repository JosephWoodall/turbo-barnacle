from machine_learning_pipeline.model import Model
from machine_learning_pipeline.model_engineering import ModelEngineering
from machine_learning_pipeline.model_evaluation import ModelEvaluation
from machine_learning_pipeline.model_packaging import ModelPackaging

import kfp
from kfp import dsl, compiler
import kfp.components as comp


class MachineLearningPipeline:

    def __init__(self):
        self.ModelEngineering = ModelEngineering
        self.ModelEvaluation = ModelEvaluation
        self.ModelPackaging = ModelPackaging
        self.Model = Model

    @dsl.component
    def _model_engineering(self):
        return self.ModelEngineering()

    @dsl.component
    def _model_evaluation(self):
        return self.ModelEvaluation()

    @dsl.component
    def _model_packaging(self):
        return self.ModelPackaging()

    @dsl.component
    def _model(self):
        return self.Model()

    @dsl.pipeline(name="MACHINE_LEARNING_PIPELINE_ml_system_boilerplate_code_pipeline",
                  description="templatized pipeline ftw",
                  )
    def run_pipeline(self):
        print("RUNNING MACHINE LEARNING PIPELINE")
        pass
        # check pre-test checks criteria here
        # with dsl.Condition():
        #   return (1, model_object)
        # check model object output is present

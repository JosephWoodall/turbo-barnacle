import kfp
from kfp import dsl, compiler
import kfp.components as comp


class MachineLearningPipeline(dsl.Pipeline):

    def __init__(self):
        from model import Model
        from model_engineering import ModelEngineering
        from model_evaluation import ModelEvaluation
        from model_packaging import ModelPackaging

        self.ModelEngineering = ModelEngineering
        self.ModelEvaluation = ModelEvaluation
        self.ModelPackaging = ModelPackaging
        self.Model = Model
        self.model_object = None

    @dsl.component
    def __model_engineering(self):
        return self.ModelEngineering()

    @dsl.component
    def __model_evaluation(self):
        return self.ModelEvaluation()

    @dsl.component
    def __model_packaging(self):
        return self.ModelPackaging()

    @dsl.component
    def __model(self):
        return self.Model()

    @dsl.component
    def _model_object(self):
        self.model_object = {
            "model_object": "",
            "test_pass": 1
        }
        print(self.model_object.values())
        return self.model_object.values()

    @dsl.pipeline
    def run_pipeline(self):
        print("--RUNNING MACHINE LEARNING PIPELINE--")
        return MachineLearningPipeline._model_object().outputs

        # check pre-test checks criteria here
        # with dsl.Condition():
        #   return (1, model_object)
        # check model object output is present


if __name__ == "__main__":
    import os
    os.chdir(r'./src/ml_system_boilerplate_code/kubeflow/machine_learning_pipeline/')
    compiler.Compiler().compile(MachineLearningPipeline.run_pipeline,
                                package_path='machine_learning_pipeline.yaml')

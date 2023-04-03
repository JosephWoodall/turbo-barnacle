import joblib
from preprocessing import ModelPreprocessing


class ModelServing:
    """ """
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.preprocessor = ModelPreprocessing()

    def predict(self, input_data):
        """

        :param input_data: 

        """
        # Preprocess input data
        preprocessed_data = self.preprocessor.preprocess(input_data)

        # Make predictions using the trained model
        predictions = self.model.predict(preprocessed_data)

        # Return predictions
        return predictions

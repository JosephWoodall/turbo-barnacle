import joblib
from preprocessing import ModelPreprocessing
import pickle
import numpy as np
from flask import Flask, request, jsonify


class ModelServingWebService:
    def __init__(self, model_path):
        self.app = Flask(__name__)
        self.model = pickle.load(open(model_path, 'rb'))

        @self.app.route('/predict', methods=['POST'])
        def predict():
            # Extract data from POST request
            data = request.json

            # Make predictions on the data
            input_data = np.array(data['input'])
            output_data = self.model.predict(input_data)

            # Format response as JSON
            response = {
                'output': output_data.tolist()
            }

            return jsonify(response)

    def run(self, port=8080):
        self.app.run(port=port)


'''
# Example usage
server = ModelServingWebService('path/to/serialized/model.pkl')
server.run()
'''

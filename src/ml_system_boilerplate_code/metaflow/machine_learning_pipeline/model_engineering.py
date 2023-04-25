'''
includes feature engineering, hyperparameter tuning
'''
import h2o
from h2o.automl import H2OAutoML
import numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class ModelEngineering:

    def __init__(self, explanatory_var=[], response_var=[]):
        print("-----MODEL ENGINEERING INITIALIZED-----")
        self.explanatory_var = explanatory_var
        self.response_var = response_var

    def _hashed_features(self):
        """
        _hashed_feature is a bucket of deterministic and portable hash of string representation and accept the trade-off of collisions in the data representation.

        This function solves problems associated with categorical features such as incomplete vocabulary, model size due to cardinality, and cold start.

        """
        hased_features = hash(frozenset(self.arg.items()))
        return hased_features

    def _embedding(self, max_sequence_length=10, embedding_dimension=64) -> dict:
        """
        _embedding learns a data representation that maps high-cardinality data into a lower dimensional space in such a way that the information relevant to the learning problem is solved.

        This function solves the problem of high-cardinality features where closeness relationships are important to preserve.
        """
        num_words = (len(self.arg.keys()) * len(self.arg))
        tokenizer = Tokenizer(num_words=num_words)
        for i in self.arg:
            tokenizer.fit_on_texts(self.arg[i])

        # Convert the text data in each column to sequences of integers and pad the sequences
        max_len = min(max_sequence_length, max(
            [len(tokenizer.texts_to_sequences(self.arg[col])) for col in self.arg]))
        for col in self.arg:
            sequences = tokenizer.texts_to_sequences(self.arg[col])
            padded_sequences = pad_sequences(
                sequences, maxlen=max_len, padding='post', truncating='post')
            embedding_matrix = numpy.random.rand(
                len(tokenizer.word_index)+1, embedding_dimension)
            self.arg[col] = numpy.concatenate([padded_sequences, numpy.zeros(
                (padded_sequences.shape[0], 1)), embedding_matrix[padded_sequences]], axis=1)
        return self.arg

    def _feature_cross(self, feature_list: list) -> dict:
        """
        _feature_cross helps models learn relationships between inputs faster by explicitly making each combination of input values a separate feature.

        This function solves model complexity insufficiency to learn feature relationships.
        """
        for i in range(len(feature_list)):
            for j in range(i + 1, len(feature_list)):
                feature1_name = feature_list[i]
                feature2_name = feature_list[j]
                feature_cross = f'{feature1_name}_{feature2_name}'
                self.arg[feature_cross] = [
                    a * b for a, b in zip(self.arg[feature1_name], self.arg[feature2_name])]
        return self.arg

    def _model_selection(self):
        """
        _model_selection tests a number of models based on training and testing metrics
        """
        h2o.init()
        aml = H2OAutoML(max_models=20, seed=1)
        aml.train(x=self.explanatory_var,
                  y=self.response_var, training_frame="")  # training_frame needs to be the train object from the DataVersioning class
        return aml.leader

    def _hyperparameter_tuning(self):
        """
        _hyperparameter_tuning optimizes hyperparameters of selected model
        """
        pass


if __name__ == '__main__':
    ModelEngineering()

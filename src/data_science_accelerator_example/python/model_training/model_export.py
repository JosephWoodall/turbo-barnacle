import joblib


class ModelExporter:
    """ """
    def __init__(self, model):
        self.model = model

    def export(self, path):
        """

        :param path: 

        """
        joblib.dump(self.model, path)

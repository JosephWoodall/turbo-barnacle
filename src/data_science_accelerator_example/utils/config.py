import yaml


class Config:
    def __init__(self, path):
        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key):
        return self._config[key]

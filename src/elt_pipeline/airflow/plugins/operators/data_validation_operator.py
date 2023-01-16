from airflow.plugins_manager import AirflowPlugin
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from great_expectations.dataset import PandasDataset

class DataValidationOperator(BaseOperator):
    @apply_defaults
    def __init__(self, validation_config_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_config_path = validation_config_path

    def execute(self, context):
        # load data
        data = pd.read_parquet("path_to_data")
        dataset = PandasDataset(data)
        # load validation config
        validation_config = load_yaml(self.validation_config_path)
        for expectation in validation_config:
            expectation_config = validation_config[expectation]
            result = dataset.validate_expectation(expectation_config)
            if result:
                raise ValueError(f"Data validation failed: {result}")

class DataValidationOperatorPlugin(AirflowPlugin):
    name = 'data_validation_operator_plugin'
    operators = [DataValidationOperator]
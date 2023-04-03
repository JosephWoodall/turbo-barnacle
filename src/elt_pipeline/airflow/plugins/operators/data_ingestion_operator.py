from airflow.plugins_manager import AirflowPlugin
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class DataIngestionOperator(BaseOperator):
    """ """
    @apply_defaults
    def __init__(self, source_type, source_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_type = source_type
        self.source_config = source_config

    def execute(self, context):
        """

        :param context: 

        """
        if self.source_type == 'rest_api':
            # code to ingest data from rest api
            pass
        elif self.source_type == 'vpn':
            # code to ingest data from vpn
            pass

class DataIngestionOperatorPlugin(AirflowPlugin):
    """ """
    name = 'data_ingestion_operator_plugin'
    operators = [DataIngestionOperator]

import os
import requests
import json
from pyspark.sql import SparkSession
import pandas as pd
import great_expectations as ge
from data_validation import DataValidation
from transformations import Transformation1, Transformation2
from utils import DataIngestion, DataStorage

class DataPipeline:
    """ """
    def __init__(self, raw_data_path, transformed_data_path, validation_config_path):
        self.raw_data_path = raw_data_path
        self.transformed_data_path = transformed_data_path
        self.validation_config_path = validation_config_path

    def ingest_data(self):
        """ """
        response = requests.get(REST_API_URL)
        data = response.json()
        DataIngestion.save_to_file(data, self.raw_data_path)

    def validate_data(self):
        """ """
        # Initialize the Expectations Suite
        self.validation = DataValidation(self.validation_config_path)
        # Load the raw data
        self.raw_data = DataStorage.load_from_file(self.raw_data_path)
        # Run the validation checks
        self.validation.run_validations(self.raw_data)

    def transform_data(self):
        """ """
        spark = SparkSession.builder.appName("DataPipeline").getOrCreate()
        df = spark.read.json(self.raw_data_path)
        # Perform transformations using PySpark and Pandas
        df = Transformation1.execute(df)
        df = Transformation2.execute(df)
        # Save the transformed data to a parquet file
        DataStorage.save_to_parquet(df, self.transformed_data_path)
        spark.stop()

    def run(self):
        """ """
        self.ingest_data()
        self.validate_data()
        self.transform_data()
from pyspark.sql import SparkSession

class DataStorage:
    @staticmethod
    def save_to_parquet(dataframe, file_path):
        # Save the dataframe to a parquet file
        dataframe.write.parquet(file_path, mode='overwrite')

    @staticmethod
    def load_from_parquet(file_path):
        # Load the dataframe from a parquet file
        spark = SparkSession.builder.appName("DataStorage").getOrCreate()
        dataframe = spark.read.parquet(file_path)
        spark.stop()
        return dataframe

    @staticmethod
    def load_from_file(file_path):
        # Load data from a file
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
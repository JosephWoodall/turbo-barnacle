class Transformation1:
    """ """
    @staticmethod
    def execute(dataframe):
        """

        :param dataframe: 

        """
        # Perform transformation 1 using PySpark and Pandas
        dataframe = dataframe.filter(dataframe.column1 > 0)
        dataframe = dataframe.withColumnRenamed("column2", "renamed_column2")
        dataframe = dataframe.groupBy("column3").agg({"column4": "mean"})
        return dataframe
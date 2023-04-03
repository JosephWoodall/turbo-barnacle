class Transformation2:
    """ """
    @staticmethod
    def execute(dataframe):
        """

        :param dataframe: 

        """
        # Perform transformation 2 using Pandas
        dataframe = dataframe.toPandas()
        dataframe["column5"] = dataframe["column5"].apply(lambda x: x.upper())
        dataframe = dataframe.dropna(subset=["column6"])
        dataframe["new_column"] = dataframe["column1"] + dataframe["column2"]
        dataframe = dataframe.groupby("column3").agg({"new_column":"mean"})
        dataframe = dataframe.rename(columns={"new_column":"mean_new_column"})
        return dataframe
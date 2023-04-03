from sqlalchemy import create_engine
import pandas as pd

class DataValidator:
    """ """
    def __init__(self, source_connection_string, target_connection_string):
        self.source_conn = create_engine(source_connection_string)
        self.target_conn = create_engine(target_connection_string)
    
    def validate_table(self, table_name):
        """

        :param table_name: 

        """
        # Retrieve data from source table
        source_data = pd.read_sql("SELECT * FROM {}".format(table_name), self.source_conn)
        # Retrieve data from target table
        target_data = pd.read_sql("SELECT * FROM {}".format(table_name), self.target_conn)
        # Perform validation on the data
        if source_data.equals(target_data):
            print("Data in {} table is valid!".format(table_name))
        else:
            print("Data in {} table is not valid.".format(table_name))

    def validate_all(self, exclude_tables=[]):
        """

        :param exclude_tables:  (Default value = [])

        """
        # Get the list of tables in the source database
        source_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", self.source_conn)
        # Get the list of tables in the target database
        target_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", self.target_conn)
        # Compare the tables in both databases
        if set(source_tables) == set(target_tables):
            print("Both source and target databases have the same set of tables.")
            for table in source_tables:
                if table not in exclude_tables:
                    self.validate_table(table)
        else:
            print("Both source and target databases have different set of tables.")
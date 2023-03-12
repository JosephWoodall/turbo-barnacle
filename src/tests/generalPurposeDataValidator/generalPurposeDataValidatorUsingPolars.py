import polars as pl
import psycopg2

class DataValidator:
    def __init__(self, conn_str):
        self.conn_str = conn_str

    def fetch_data(self, query):
        with psycopg2.connect(self.conn_str) as conn:
            df = pl.from_postgres(conn, query)
        return df

    def read_csv(self, file_path):
        df = pl.read_csv(file_path)
        return df

    def validate_column(self, col_name, df1, df2):
        is_equal = df1[col_name] == df2[col_name]
        return is_equal

    def validate_datasets(self, common_col, df1, df2):
        # check if column is present in both datasets
        if common_col not in df1.columns or common_col not in df2.columns:
            raise ValueError(f"Column '{common_col}' not found in both datasets.")

        # check if common column values match
        is_equal = self.validate_column(common_col, df1, df2)
        if not is_equal.all():
            raise ValueError(f"Data in column '{common_col}' does not match between datasets.")
        
        # additional validation tests can be added here
        # ...

        return True


'''
Example Usage 

# fetching data from a database
conn_str = "postgresql://user:password@host:port/database"
validator = DataValidator(conn_str)

query = "SELECT * FROM my_table"
df1 = validator.fetch_data(query)

# fetching data frmo a csv file
file_path = "my_file.csv"
df2 = validator.read_csv(file_path)

# specify the common column
common_col = "my_common_column"
is_valid = validator.validate_datasets(common_col, df1, df2)

'''
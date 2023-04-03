import psycopg2
import csv

class DataValidator:
    """ """
    def __init__(self, conn_str):
        self.conn_str = conn_str

    def fetch_data(self, query):
        """

        :param query: 

        """
        with psycopg2.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                data = [dict(zip(columns, row)) for row in rows]
        return data

    def read_csv(self, file_path):
        """

        :param file_path: 

        """
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
        return data

    def validate_column(self, col_name, data1, data2):
        """

        :param col_name: param data1:
        :param data2: param data1:
        :param data1: 

        """
        is_equal = [row1[col_name] == row2[col_name] for row1, row2 in zip(data1, data2)]
        return is_equal

    def validate_datasets(self, common_col, data1, data2):
        """

        :param common_col: param data1:
        :param data2: param data1:
        :param data1: 

        """
        # check if column is present in both datasets
        if common_col not in data1[0].keys() or common_col not in data2[0].keys():
            raise ValueError(f"Column '{common_col}' not found in both datasets.")

        # check if common column values match
        is_equal = self.validate_column(common_col, data1, data2)
        if not all(is_equal):
            raise ValueError(f"Data in column '{common_col}' does not match between datasets.")

        # group the data by the common column
        groups1 = {}
        for row in data1:
            key = row[common_col]
            if key in groups1:
                groups1[key].append(row)
            else:
                groups1[key] = [row]
        groups2 = {}
        for row in data2:
            key = row[common_col]
            if key in groups2:
                groups2[key].append(row)
            else:
                groups2[key] = [row]

        # calculate the difference between the financial columns of your choice (revenue, margin, and cost columns, for example)
        diff_dict = {}
        for key in groups1.keys() & groups2.keys():
            diff_dict[key] = {col: sum(row[col] for row in groups1[key]) - sum(row[col] for row in groups2[key]) for col in ['FINANCIAL_COLUMN_1', 'FINANCIAL_COLUMN_2', 'FINANCIAL_COLUMN_3']}

        # display the difference between the financial columns of your choice (revenue, margin, and cost columns for each common customer for example)
        print(diff_dict)

        return True


'''
Example Usage 

# fetching data from a database
conn_str = "postgresql://user:password@host:port/database"
validator = DataValidator(conn_str)

query = "SELECT * FROM my_table"
data1 = validator.fetch_data(query)

# fetching data from a csv file
file_path = "my_file.csv"
data2 = validator.read_csv(file_path)

# specify the common column
common_col = "my_common_column"
is_valid = validator.validate_datasets(common_col, data1, data2)
'''
import psycopg2
import csv

class DataValidator:
    def __init__(self, conn_str):
        self.conn_str = conn_str

    def fetch_data(self, query):
        with psycopg2.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                data = cur.fetchall()
                headers = [desc[0] for desc in cur.description]
                return self.create_dict(headers, data)

    def read_csv(self, file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)
            return self.create_dict(headers, data)

    def create_dict(self, headers, data):
        data_dict = []
        for row in data:
            row_dict = {}
            for i, h in enumerate(headers):
                row_dict[h] = row[i]
            data_dict.append(row_dict)
        return data_dict

    def validate_column(self, col_name, data1, data2):
        is_equal = [d1[col_name] == d2[col_name] for d1, d2 in zip(data1, data2)]
        return is_equal

    def validate_datasets(self, common_col, data1, data2):
        # check if column is present in both datasets
        if common_col not in data1[0] or common_col not in data2[0]:
            raise ValueError(f"Column '{common_col}' not found in both datasets.")

        # check if common column values match
        is_equal = self.validate_column(common_col, data1, data2)
        if not all(is_equal):
            raise ValueError(f"Data in column '{common_col}' does not match between datasets.")
        
        # additional validation tests can be added here
        # ...

        return True

'''
Example Usage:

# create DataValidator object with database connection string
validator = DataValidator('dbname=mydb user=myuser password=mypass host=localhost port=5432')

# fetch data from database
data1 = validator.fetch_data('SELECT * FROM table1')
data2 = validator.fetch_data('SELECT * FROM table2')

# read CSV files
data1 = validator.read_csv('file1.csv')
data2 = validator.read_csv('file2.csv')

'''
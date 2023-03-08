import csv
import xlrd
import psycopg2
import cx_Oracle
import pyodbc
import os

import unittest 

class DataReader:
    def __init__(self, source_type=None, connection_string=None):
        self.source_type = source_type
        self.connection_string = connection_string
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")

    def read_data(self, query):
        if not self.source_type:
            self.source_type = os.getenv("DB_SOURCE_TYPE")

        if not self.connection_string:
            self.connection_string = f"host={self.db_host} port={self.db_port} dbname={self.db_name} user={self.db_user} password={self.db_password}"

        if self.source_type.lower() == 'csv':
            with open(self.connection_string, newline='') as csvfile:
                reader = csv.reader(csvfile)
                data = list(reader)
        elif self.source_type.lower() == 'excel':
            book = xlrd.open_workbook(self.connection_string)
            sheet = book.sheet_by_index(0)
            data = []
            for row_idx in range(0, sheet.nrows):
                row = []
                for col_idx in range(0, sheet.ncols):
                    cell_value = sheet.cell(row_idx, col_idx).value
                    row.append(cell_value)
                data.append(row)
        elif self.source_type.lower() == 'postgresql':
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            cursor.close()
            conn.close()
        elif self.source_type.lower() == 'oracle':
            conn = cx_Oracle.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            cursor.close()
            conn.close()
        elif self.source_type.lower() == 'sql server':
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            cursor.close()
            conn.close()
        else:
            raise ValueError('Invalid data source type')

        return data

class DataValidator(unittest.TestCase):
    def __init__(self, data_objects):
        super().__init__()
        self.data_objects = data_objects
        
    '''
    Compares multiple data objects
    '''
    def assert_data_equal(self):
        for i in range(len(self.data_objects) -1):
            for j in range(i + 1, len(self.data_objects)):
                self.assertEqual(self.data_objects[i], self.data_objects[j])

    def assert_column_names_equal(self):
        column_names = [set(data.columns) for data in self.data_objects]
        for i in range(len(column_names) - 1):
            for j in range(i + 1, len(column_names)):
                self.assertSetEqual(column_names[i], column_names[j])       

    def assert_data_length_equal(self):
        for i in range(len(self.data_objects) -1):
            for j in range(i + 1, len(self.data_objects)):
                self.assertEqual(len(self.data_objects[i]), len(self.data_objects[j]))
    
    def assert_data_elements_equal(self):
        for i in range(len(self.data_objects) -1):
            for j in range(i + 1, len(self.data_objects)):
                self.assertCountEqual(self.data_objects[i], self.data_objects[j])
    
    def assert_data_structure_equal(self):
        for i in range(len(self.data_objects) -1):
            for j in range(i + 1, len(self.data_objects)):
                self.assertDictEqual(self.data_objects[i], self.data_objects[j])
    
    '''
    Compares only a single data object, one at a time, for each data object
    '''
    
    '''
    To include the below tests: 
        Testing if the data is in the expected format:

        Test if the data is in the expected file format (e.g., CSV, JSON, XML).
        Test if the data contains the expected fields or columns.
        Test if the data uses the expected data types (e.g., strings, integers, floats).
        Testing if the data contains valid values:

        Test if the data contains only valid values (e.g., no missing or null values).
        Test if the data contains values within expected ranges or thresholds.
        Test if the data contains values that are consistent with domain knowledge.
        Testing if the data is complete:

        Test if the data contains all the expected records or observations.
        Test if the data contains all the expected fields or columns.
        Test if the data contains all the expected categories or classes.
        Testing if the data is consistent:

        Test if the data is consistent across multiple sources or versions.
        Test if the data is consistent within itself (e.g., no contradictory or duplicated records).
        Test if the data is consistent with external sources or benchmarks.
        Testing if the data is usable:

        Test if the data can be loaded and accessed efficiently.
        Test if the data can be analyzed and visualized as expected.
        Test if the data can be used for the intended purpose (e.g., training a machine learning model).
    '''
    
    
    
'''
Example Usage in a main.py file: 

data_reader = DataReader()

# Read data from two different tables with different column names
query1 = "SELECT column1, column2 FROM table1"
query2 = "SELECT column3, column4 FROM table2"
data1 = data_reader.read_data(query1)
data2 = data_reader.read_data(query2)
validator = DataValidator([data1, data2])

# Assert that the data objects are equal
validator.assert_data_equal()

# Assert that the column names are equal
validator.assert_column_names_equal()


# Read data from two different tables with same column names
query3 = "SELECT column1, column2 FROM table3"
query4 = "SELECT column1, column2 FROM table4"
data3 = data_reader.read_data(query3)
data4 = data_reader.read_data(query4)
validator = DataValidator([data3, data4])

# Assert that the data objects are equal
validator.assert_data_equal()

# Assert that the column names are equal
validator.assert_column_names_equal()

if __name__ == '__main__':
    unittest.main()

'''
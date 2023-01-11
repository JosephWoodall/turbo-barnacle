import pyodbc
import pandas as pd

class MSSQLExtractor:
    def __init__(self, server, database, username, password):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.cnxn = None
        self.cursor = None
        
    def connect(self):
        # Connect to the database
        self.cnxn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}')
        self.cursor = self.cnxn.cursor()
        
    def extract_data(self, query):
        # Execute the query and fetch the data
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return rows
    
    def close_connection(self):
        # Close the cursor and connection
        self.cursor.close()
        self.cnxn.close()

# usage of the class
mssql = MSSQLExtractor('server', 'database', 'username', 'password')
mssql.connect()
data = mssql.extract_data('SELECT * FROM table_name')
print(data)
mssql.close_connection()
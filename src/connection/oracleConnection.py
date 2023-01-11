import cx_Oracle
import pandas as pd

class OracleDataExtractor:
    def __init__(self, user, password, host, port, sid):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.sid = sid
        self.connection = None
        self.cursor = None
        
    def connect(self):
        # Connect to Oracle database
        self.connection = cx_Oracle.connect(f'{self.user}/{self.password}@{self.host}:{self.port}/{self.sid}')
        self.cursor = self.connection.cursor()
    
    def extract_data(self, table_name):
        # Define the SQL query to extract data
        query = f'SELECT * FROM {table_name}'
        
        # Execute the query and fetch the data
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        return data
    
    def close_connection(self):
        # Close the cursor and connection
        self.cursor.close()
        self.connection.close()
        
    def preprocess_data(self, data):
        # Transform the data as needed
        # (for example, you can use Pandas to perform data cleaning, normalization, etc.)
        data = pd.DataFrame(data)
        return data

# usage of the class
data_extractor = OracleDataExtractor('user', 'password', 'host', 'port', 'sid')
data_extractor.connect()
data = data_extractor.extract_data('table_name')
data = data_extractor.preprocess_data(data)
print(data)
data_extractor.close_connection()
import pandas as pd
import sqlite3
import psycopg2
import pyodbc
import unittest


class DataComparer:
    def __init__(self, source=None, target=None):
        self.source = source
        self.target = target

    def load_data_from_csv(self, file_path):
        return pd.read_csv(file_path)

    def connect_to_database(self, db_type, db_name, host, port, user, password, table_name):
        if db_type == 'sqlite':
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        elif db_type == 'postgres':
            conn = psycopg2.connect(
                host=host, port=port, user=user, password=password, database=db_name)
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        elif db_type == 'oracle':
            dsn_tns = f"(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST={host})(PORT={port})))(CONNECT_DATA=(SID={db_name})))"
            conn = cx_Oracle.connect(user=user, password=password, dsn=dsn_tns)
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        elif db_type == 'sqlserver':
            conn = pyodbc.connect(
                f"Driver={{SQL Server}};Server={host};Database={db_name};UID={user};PWD={password}")
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        else:
            raise ValueError(f"{db_type} is not a valid database type.")


class TestDataComparer(unittest.TestCase):
    def setUp(self):
        self.comparer = DataComparer(source={'database_name': 'database1', 'host': 'localhost', 'port': 5432, 'user': 'user1', 'password': 'password1', 'table_name': 'table1'},
                                     target={'database_name': 'database2', 'host': 'localhost', '
    def test_compare_data_when_source_data_is_none(self):
        self.source_data=None
        with self.assertRaises(Exception):
            self.comparer.compare_data(self.source_data, self.target_data)

    def test_compare_data_when_target_data_is_none(self):
        self.target_data=None
        with self.assertRaises(Exception):
            self.comparer.compare_data(self.source_data, self.target_data)

    def test_compare_data_when_source_and_target_data_are_none(self):
        self.source_data=None
        self.target_data=None
        with self.assertRaises(Exception):
            self.comparer.compare_data(self.source_data, self.target_data)

    def test_compare_data_when_columns_are_different(self):
        self.source_data=pd.DataFrame(
            {'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 40]})
        self.target_data=pd.DataFrame({'id': [1, 2, 3], 'name': [
                                      'Alice', 'Bob', 'Charlie'], 'address': ['NY', 'LA', 'SF']})
        with self.assertRaises(Exception):
            self.comparer.compare_data(self.source_data, self.target_data)

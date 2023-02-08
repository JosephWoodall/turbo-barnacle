import cx_Oracle
import psycopg2
import time

"""
source_type = "oracle"
target_type = "postgresql"

source_conn_params = {
    "user": "user",
    "password": "password",
    "host": "host",
    "port": "port",
    "service_name": "service_name"
}

target_conn_params = {
    "user": "user",
    "password": "password",
    "host": "host",
    "port": "port",
    "database": "database"
}
"""

class DatabaseMigration:
    
    def __init__(self, source_type, target_type, source_conn_params, target_conn_params):
        self.source_type = source_type
        self.target_type = target_type
        self.source_conn_params = source_conn_params
        self.target_conn_params = target_conn_params
        self.total_rows = 0
        self.data = []
        
    def connect_to_source(self):
        if self.source_type == "oracle":
            conn = cx_Oracle.connect(self.source_conn_params)
        elif self.source_type == "postgresql":
            conn = psycopg2.connect(self.source_conn_params)
        else:
            raise Exception("Unsupported source database type")
        return conn

    def connect_to_target(self):
        if self.target_type == "oracle":
            conn = cx_Oracle.connect(self.target_conn_params)
        elif self.target_type == "postgresql":
            conn = psycopg2.connect(self.target_conn_params)
        else:
            raise Exception("Unsupported target database type")
        return conn

    def get_data_from_source(self):
        source_conn = self.connect_to_source()
        source_cursor = source_conn.cursor()
        
        if self.source_type == "oracle":
            source_cursor.execute("SELECT COUNT(*) FROM source_table")
            self.total_rows = source_cursor.fetchone()[0]
            source_cursor.execute("SELECT * FROM source_table")
            self.data = source_cursor.fetchall()
            print("\n" + "="*40)
            print("\tData retrieval complete")
            print("="*40)
            print(f"\tTotal rows retrieved: {self.total_rows}")
            print("="*40 + "\n")
        elif self.source_type == "postgresql":
            source_cursor.execute("SELECT COUNT(*) FROM source_table")
            self.total_rows = source_cursor.fetchone()[0]
            source_cursor.execute("SELECT * FROM source_table")
            self.data = source_cursor.fetchall()
            print("\n" + "="*40)
            print("\tData retrieval complete")
            print("="*40)
            print(f"\tTotal rows retrieved: {self.total_rows}")
            print("="*40 + "\n")
        else:
            raise Exception("Unsupported source database type")
        
        source_cursor.close()
        source_conn.close()

        
    def insert_data_into_target(self):
        target_conn = self.connect_to_target()
        target_cursor = target_conn.cursor()

        if self.target_type == "oracle":
            start_time = time.time()
            for row in self.data:
                target_cursor.execute("INSERT INTO target_table VALUES (:1, :2, ...)", row)
            target_conn.commit()
            print("\n" + "="*40)
            print("\tData migration complete")
            print("="*40)
            print(f"\tTime taken: {time.time() - start_time:.2f} seconds")
            print("="*40 + "\n")
        elif self.target_type == "postgresql":
            start_time = time.time()
            target_cursor.executemany("INSERT INTO target_table VALUES (%s, %s, ...)", self.data)
            target_conn.commit()
            print("\n" + "="*40)
            print("\tData migration complete")
            print("="*40)
            print(f"\tTime taken: {time.time() - start_time:.2f} seconds")
            print("="*40 + "\n")
        else:
            raise Exception("Unsupported target database type")
        
        target_cursor.close()
        target_conn.close()

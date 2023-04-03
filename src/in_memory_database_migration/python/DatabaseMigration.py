import psycopg2
import cx_Oracle

class DatabaseMigration:
    """ """
    def __init__(self, source_type, target_type, source_conn_params, target_conn_params):
        self.source_type = source_type
        self.target_type = target_type
        self.source_conn_params = source_conn_params
        self.target_conn_params = target_conn_params

    def establish_connection(self, db_type, conn_params):
        """

        :param db_type: param conn_params:
        :param conn_params: 

        """
        if db_type == "oracle":
            conn = cx_Oracle.connect(conn_params["user"], conn_params["password"], conn_params["host"])
        elif db_type == "postgresql":
            conn = psycopg2.connect(
                host=conn_params["host"],
                database=conn_params["database"],
                user=conn_params["user"],
                password=conn_params["password"]
            )
        else:
            raise ValueError("Invalid database type")
        return conn

    def fetch_data_from_source(self, source_conn, source_table):
        """

        :param source_conn: param source_table:
        :param source_table: 

        """
        source_cursor = source_conn.cursor()
        source_cursor.execute(f"SELECT * FROM {source_table}")
        return source_cursor.fetchall()

    def insert_data_into_target(self, target_conn, target_table, data):
        """

        :param target_conn: param target_table:
        :param data: param target_table:
        :param target_table: 

        """
        target_cursor = target_conn.cursor()
        for i, row in enumerate(data):
            target_cursor.execute(f"INSERT INTO {target_table} VALUES {row}")
            print(f"{i + 1} rows inserted")
        target_conn.commit()

    def migrate_data(self, source_table, target_table):
        """

        :param source_table: param target_table:
        :param target_table: 

        """
        source_conn = self.establish_connection(self.source_type, self.source_conn_params)
        target_conn = self.establish_connection(self.target_type, self.target_conn_params)

        data = self.fetch_data_from_source(source_conn, source_table)
        self.insert_data_into_target(target_conn, target_table, data)

        source_conn.close()
        target_conn.close()

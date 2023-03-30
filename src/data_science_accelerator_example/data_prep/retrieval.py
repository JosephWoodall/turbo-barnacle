import polars as pl
import os


class Retrieval:
    def __init__(self, source):
        self.source = source
        self.credentials = self._get_credentials()

    def _get_credentials(self):
        db_type = self.source.split(':')[0]
        username = os.environ[f"{db_type.upper()}_USERNAME"]
        password = os.environ[f"{db_type.upper()}_PASSWORD"]
        host = os.environ[f"{db_type.upper()}_HOST"]
        port = os.environ[f"{db_type.upper()}_PORT"]
        database = os.environ[f"{db_type.upper()}_DATABASE"]
        return {'username': username, 'password': password, 'host': host, 'port': port, 'database': database}

    def get_data(self):
        if self.source.startswith('postgresql'):
            return self._get_postgresql_data()
        elif self.source.startswith('oracle'):
            return self._get_oracle_data()
        elif self.source.startswith('sqlserver'):
            return self._get_sqlserver_data()
        else:
            raise ValueError(f"Unsupported source: {self.source}")

    def _get_postgresql_data(self):
        conn_str = f"postgresql://{self.credentials['username']}:{self.credentials['password']}@{self.credentials['host']}:{self.credentials['port']}/{self.credentials['database']}"
        return pl.read_sql(conn_str, self.source)

    def _get_oracle_data(self):
        conn_str = f"oracle://{self.credentials['username']}:{self.credentials['password']}@{self.credentials['host']}:{self.credentials['port']}/{self.credentials['database']}"
        return pl.read_sql(conn_str, self.source)

    def _get_sqlserver_data(self):
        conn_str = f"mssql://{self.credentials['username']}:{self.credentials['password']}@{self.credentials['host']}:{self.credentials['port']}/{self.credentials['database']}"
        return pl.read_sql(conn_str, self.source)

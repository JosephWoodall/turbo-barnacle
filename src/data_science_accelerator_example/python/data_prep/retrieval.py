import polars as pl
import os


class Retrieval:
    """
     The Retrieval class is a method to retrieve data from a cloud server.
    """

    def __init__(self, source):
        self.source = source
        self.credentials = self._get_credentials()

    def _get_credentials(self) -> dict:
        """
        _get_credentials retrieves database credentials from an environment file used to connect to the database.

        Returns:
            dict: username, password, host, port, and database key value pairs.
        """
        db_type = self.source.split(':')[0]
        username = os.environ[f"{db_type.upper()}_USERNAME"]
        password = os.environ[f"{db_type.upper()}_PASSWORD"]
        host = os.environ[f"{db_type.upper()}_HOST"]
        port = os.environ[f"{db_type.upper()}_PORT"]
        database = os.environ[f"{db_type.upper()}_DATABASE"]
        return {'username': username, 'password': password, 'host': host, 'port': port, 'database': database}

    def get_data(self) -> callable[pl.DataFrame]:
        """
        get_data performs the query process and retrieval of the data.

        Raises:
            ValueError: specifies the incorrectly passed source type. Note: the supported source types are Postgresql, Oracle, and SQL Server.

        Returns:
            callable: getter functions
        """
        if self.source.startswith('postgresql'):
            return self._get_postgresql_data()
        elif self.source.startswith('oracle'):
            return self._get_oracle_data()
        elif self.source.startswith('sqlserver'):
            return self._get_sqlserver_data()
        else:
            raise ValueError(f"Unsupported source: {self.source}")

    def _get_postgresql_data(self) -> pl.DataFrame:
        """
        _get_postgresql_data getter for data from a Postgresql database

        Returns:
            pl.DataFrame: Postgresql database data stored into a polars Dataframe.
        """
        conn_str = f"postgresql://{self.credentials['username']}:{self.credentials['password']}@{self.credentials['host']}:{self.credentials['port']}/{self.credentials['database']}"
        return pl.read_sql(conn_str, self.source)

    def _get_oracle_data(self) -> pl.DataFrame:
        """
        _get_oracle_data getter for data from an Oracle database.

        Returns:
            pl.DataFrame: Oracle database data stored into a polars Dataframe.
        """
        conn_str = f"oracle://{self.credentials['username']}:{self.credentials['password']}@{self.credentials['host']}:{self.credentials['port']}/{self.credentials['database']}"
        return pl.read_sql(conn_str, self.source)

    def _get_sqlserver_data(self) -> pl.DataFrame:
        """
        _get_sqlserver_data getter for data from a SQL Server database.

        Returns:
            pl.DataFrame: SQL Server database data stored into a polars Dataframe.
        """
        conn_str = f"mssql://{self.credentials['username']}:{self.credentials['password']}@{self.credentials['host']}:{self.credentials['port']}/{self.credentials['database']}"
        return pl.read_sql(conn_str, self.source)

from in_memory_database_migration import DatabaseMigration

# Specify the source and target database types
source_type = "oracle"
target_type = "postgresql"

# Specify the connection parameters for the source and target databases
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

# Create an instance of the DatabaseMigration class
db_migration = DatabaseMigration(source_type, target_type, source_conn_params, target_conn_params)

# Fetch the data from the source database
data = db_migration.fetch_data_from_source("source_table_name")

# Insert the data into the target database
db_migration.insert_data_into_target("target_table_name", data)

# Close the connections to the source and target databases
db_migration.close_connections()
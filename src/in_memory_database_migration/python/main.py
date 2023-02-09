from in_memory_database_migration import DatabaseMigration

source_conn_params = {
    "host": "hostname_or_ip",
    "database": "database_name",
    "user": "username",
    "password": "password"
}

target_conn_params = {
    "host": "hostname_or_ip",
    "database": "database_name",
    "user": "username",
    "password": "password"
}

migration = DatabaseMigration("oracle", "postgresql", source_conn_params, target_conn_params)
migration.migrate_data("source_table", "target_table")

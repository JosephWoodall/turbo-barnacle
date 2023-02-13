class DatabaseMigration {
  constructor(source, target) {
    this.source = source; 
    this.target = target;
  }

  extract() {
    switch(this.source.type){
      case 'postgresql':
        return this.extractFromPostgresql();
      case 'mongodb':
        return this.extractFromMongodb();
      case 'oracle':
        return this.extractFromOracle();
      case 'sql_server':
        return this.extractFromSqlServer();
      default: 
        throw new Error('Unsupported source database type: ${this.source.type}');
    }
  }

  extractFromPostgresql() {
    // Implementation for extracting data from a PostgreSQL database
  }

  extractFromMongodb() {
    // Implementation for extracting data from a MongoDB database
  }

  extractFromOracle() {
    // Implementation for extracting data from an Oracle database
  }

  extractFromSqlServer() {
    //Implementation for extracting data from a SQL Server database
  }

  transform(data) {
    // Placeholder for any data transformations
    return data;
  }

  load(data) {
    switch (this.target.type) {
      case 'postgresql': 
        return this.loadIntoPostgresql(data);
      case 'mongodb': 
        return this.loadIntoMongoDb(data);
      case 'oracle': 
        return this.loadIntoOracle(data);
      case 'sql_server':
        return this.loadIntoSqlServer(data);
      default: 
        throw new Error('Unsupported target database type: ${this.target.type}');
    }
  }

  loadIntoPostgresql(data) {
    // Implementation for loading data into a PostgreSQL database
  }

  loadIntoMongoDb(data) {
    // Implementation for loading data into a MongoDB database
  }

  loadIntoOracle(data) {
    // Implementation for loading data into an Oracle database
  }

  loadIntoSqlServer(data) {
    // Implementation for loading data into a SQL Server database
  }

  perform() {
    const data = this.extract(); 
    const transformedData = this.transform(data);
    this.load(transformedData);
  }
}

// Example usage: 
const etl = new ETL({ type: 'postgresql'}, { type: 'oracle'});
etl.perform();
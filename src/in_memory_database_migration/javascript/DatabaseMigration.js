const { Pool } = require('pg');

class DatabaseMigration {
  constructor(source, target, batchSize = 1000) {
    this.source = source; 
    this.target = target;
    this.batchSize = batchSize;
    this.sourcePool = new Pool(this.source);
    this.targetPool = new Pool(this.target);
  }

  async extract() {

    let data = [];
    const client = await this.sourcePool.connect();
    try { 
      const result = await client.query('SELECT * FROM source_table');
      data = result.rows; 
    } finally {
      client.release()
    }
    return data; 
  }

  transform(data) {
    // Placeholder for any data transformations
    return data;
  }

  async load(data) {
    let index = 0;
    while (index < data.length) {
      const batch = data.slice(index, index + this.batchSize);
      const values = batch.map((row) => `(${row.id}, '${row.name}')`).join(',');
      const client = await this.targetPool.connect();
      try {
        await client.query(`INSERT INTO target_table (id, name) VALUES ${values}`);
      } finally {
        client.release();
      }
      index += this.batchSize;
    }
  }

  async perform() {
    const data = this.extract(); 
    const transformedData = this.transform(data);
    this.load(transformedData);
  }
}

// Example usage: 

const source = {
  user: process.env.SOURCE_DB_USER,
  host: process.env.SOURCE_DB_HOST,
  database: process.env.SOURCE_DB_NAME,
  password: process.env.SOURCE_DB_PASSWORD,
  port: process.env.SOURCE_DB_PORT,
};

const target = {
  user: process.env.TARGET_DB_USER,
  host: process.env.TARGET_DB_HOST,
  database: process.env.TARGET_DB_NAME,
  password: process.env.TARGET_DB_PASSWORD,
  port: process.env.TARGET_DB_PORT,
};

const batchSize = process.env.BATCH_SIZE || 1000;

const etl = new DatabaseMigration(source, target, batchSize);
etl.perform();
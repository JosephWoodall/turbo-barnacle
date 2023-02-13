require('dotenv').config();

const pg = require('pg');
const mongoose = require('mongoose');
const oracledb = require('oracledb');
const tedious = require('tedious');
const async = require('async');

class DataValidator {
  constructor(sourceAdapter, targetAdapter) {
    this.sourceAdapter = sourceAdapter;
    this.targetAdapter = targetAdapter;
  }

  async connectSource(config) {
    await this.sourceAdapter.connect(config);
  }

  async connectTarget(config) {
    await this.targetAdapter.connect(config);
  }

  async getSourceData() {
    return await this.sourceAdapter.getData();
  }

  async getTargetData() {
    return await this.targetAdapter.getData();
  }

  
  async validateData() {
    const sourceData = await this.getSourceData();
    const targetData = await this.getTargetData();

    // perform data validation tests
    const tests = [
      // ...
    ];

    const results = [];
    for (const test of tests) {
      results.push(await test(sourceData, targetData));
    }

    return results;
  }

  async closeSource() {
    await this.sourceAdapter.close();
  }

  async closeTarget() {
    await this.targetAdapter.close();
  }

  getDbAdapter(dbType) {
    switch (dbType) {
      case 'postgresql':
        return new PostgresqlAdapter();
      case 'mongodb':
        return new MongoDbAdapter();
      case 'oracle':
        return new OracleAdapter();
      case 'sqlserver':
        return new SqlServerAdapter();
      default:
        throw new Error(`Unsupported database type: ${dbType}`);
    }
  }
}

class DbAdapter {
    async connect(config) {
        throw new Error("connect method must be implemented by a subclass");
    }
    async getData() {
        throw new Error("getDatamethod must be implemented by a subclass");
    }
    async close() {
        throw new Error("close method must be implemented by a subclass");
    }
}

class PostgresqlAdapter extends DbAdapter {
    async connect(config) {
        // connect to postgresql database
    }
    async getData() {
        // get data from postgresql database
    }
    async close() {
        // close connection to the postgresql database
    }
}

class MongoDbAdapter extends DbAdapter {
    async connect(config) {
        // connect to the MongoDB database
    }
    async getData() {
        // get data from the mongodb database
    }
    async close() {
        // close connection to the mongodb database
    }
}

class OracleAdapter extends DbAdapter {
  async connect(config) {
    // connect to Oracle database
    // ...
  }

  async getData() {
    // get data from Oracle database
    // ...
  }

  async close() {
    // close connection to Oracle database
    // ...
  }
}

class SqlServerAdapter extends DbAdapter {
    async connect(config) {
      // connect to SQL Server database
      // ...
    }
  
    async getData() {
      // get data from SQL Server database
      // ...
    }
  
    async close() {
      // close connection to SQL Server database
      // ...
    }
  }


// Example usage:

const sourceDbConfig = {
    user: process.env.SOURCE_DB_USER,
    host: process.env.SOURCE_DB_HOST,
    database: process.env.SOURCE_DB_NAME,
    password: process.env.SOURCE_DB_PASSWORD,
    port: process.env.SOURCE_DB_PORT,
    uri: process.env.SOURCE_DB_URI,
}
const targetDbConfig = {
    user: process.env.TARGET_DB_USER,
    host: process.env.TARGET_DB_HOST,
    database: process.env.TARGET_DB_NAME,
    password: process.env.TARGET_DB_PASSWORD,
    port: process.env.TARGET_DB_PORT,
    uri: process.env.TARGET_DB_URI,
}

const dataValidator = new DataValidator(new PostgresqlAdapter(), new MongoDbAdapter());

await dataValidator.connectSource(sourceDbConfig)
await dataValidator.connectTarget(targetDbConfig)

const results = await dataValidator.validateData();

console.log(results);

await dataValidator.closeSource();
await dataValidator.closeTarget();





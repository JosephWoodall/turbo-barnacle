require('dotenv').config();

const pg = require('pg');
const mongoose = require('mongoose');
const oracledb = require('oracledb');
const tedious = require('tedious');
const async = require('async');

class DataValidator {
  constructor(sourceDbType, targetDbType, sourceDbConfig, targetDbConfig) {
    this.sourceDbType = sourceDbType || process.env.SOURCE_DB_TYPE;
    this.targetDbType = targetDbType || process.env.TARGET_DB_TYPE;
    this.sourceDbConfig = sourceDbConfig;
    this.targetDbConfig = targetDbConfig;
  }

  async validate() {
    let results = [];

    try {
      let sourceDbClient, targetDbClient;
      if (this.sourceDbType === 'postgresql') {
        sourceDbClient = new pg.Client(this.sourceDbConfig);
        await sourceDbClient.connect();
      } else if (this.sourceDbType === 'mongodb') {
        mongoose.connect(this.sourceDbConfig.uri, {
          useNewUrlParser: true,
          useUnifiedTopology: true,
        });
        sourceDbClient = mongoose.connection;
      }

      if (this.targetDbType === 'postgresql') {
        targetDbClient = new pg.Client(this.targetDbConfig);
        await targetDbClient.connect();
      } else if (this.targetDbType === 'mongodb') {
        mongoose.connect(this.targetDbConfig.uri, {
          useNewUrlParser: true,
          useUnifiedTopology: true,
        });
        targetDbClient = mongoose.connection;
      } else if (this.targetDbType === 'oracle') {
        targetDbClient = await oracledb.getConnection(this.targetDbConfig);
      } else if (this.targetDbType === 'sqlserver') {
        targetDbClient = new tedious.Connection(this.targetDbConfig);
        await targetDbClient.connect();
      }

      if (this.sourceDbType === 'postgresql' && this.targetDbType === 'postgresql') {
        let sourceDbResult = await sourceDbClient.query(`SELECT COUNT(*) FROM users`);
        let targetDbResult = await targetDbClient.query(`SELECT COUNT(*) FROM users`);
        results.push({
          test: 'Number of rows in users table (source vs target)',
          result: sourceDbResult.rows[0].count === targetDbResult.rows[0].count,
        });

        sourceDbResult = await sourceDbClient.query(`SELECT COUNT(*) FROM posts`);
        targetDbResult = await targetDbClient.query(`SELECT COUNT(*) FROM posts`);
        results.push({
          test: 'Number of rows in posts table (source vs target)',
          result: sourceDbResult.rows[0].count === targetDbResult.rows[0].count,
        });
      } else if (this.sourceDbType === 'mongodb' && this.targetDbType === 'mongodb') {
        let sourceDbResult = await sourceDbClient.db.collection('users').countDocuments();
        let targetDbResult = await targetDbClient.db.collection('users').countDocuments();
        results.push({
          test: 'Number of documents in users collection (source vs target)',
          result: sourceDbResult === targetDbResult,
        });

        sourceDbResult = await sourceDbClient.db.collection('posts').countDocuments();
        targetDbResult = await targetDbClient.db.collection('posts').countDocuments();
        results.push({
          test: 'Number of documents in posts collection (source vs target)',
          result: sourceDbResult === targetDbResult,
        });
      } else {
        // Additional code to handle cross-DB validation can be added here.
      }
    } catch (err) {
      console.error(err);
    } finally {
      if (this.sourceDbType === 'postgresql') {
        sourceDbClient.end();
      } else if (this.sourceDbType === 'mongodb') {
        mongoose.disconnect();
      }
      if (this.targetDbType === 'postgresql') {
        targetDbClient.end();
      } else if (this.targetDbType === 'mongodb') {
        mongoose.disconnect();
      } else if (this.targetDbType === 'oracle') {
        targetDbClient.close();
      } else if (this.targetDbType === 'sqlserver') {
        targetDbClient.close();
      }
    }
    return results;
  }
}

// Example usage:

const sourceDbType = ""

const targetDbType = ""

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

const dataValidator = new DataValidator(sourceDbType, targetDbType, sourceDbConfig, targetDbConfig);

const results = dataValidator.validate();
console.log(results);


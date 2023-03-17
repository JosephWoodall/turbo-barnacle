# Create an IAM role for the Glue job
resource "aws_iam_role" "glue_role" {
  name = "glue_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      }
    ]
  })
}

# Attach the necessary policies to the Glue role
resource "aws_iam_role_policy_attachment" "glue_policy" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
  role = aws_iam_role.glue_role.name
}

# Create a Glue database
resource "aws_glue_catalog_database" "database" {
  name = var.glue_database_name
}

# Create a Glue table
resource "aws_glue_catalog_table" "table" {
  name = var.glue_table_name
  database_name = aws_glue_catalog_database.database.name
  table_type = "EXTERNAL_TABLE"
  parameters = {
    "classification" = "parquet"
    "typeOfData" = "file"
  }
  storage_descriptor {
    location = var.glue_table_s3_path
    input_format = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
    serde_info {
      name = "my-serde"
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
      parameters = {
        "serialization.format" = "1"
      }
    }
    columns {
      name = "column1"
      type = "string"
    }
    columns {
      name = "column2"
      type = "int"
    }
    columns {
      name = "column3"
      type = "boolean"
    }
  }
}

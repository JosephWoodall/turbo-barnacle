data "terraform_remote_state" "mlops_infra" {
  backend = "s3"
  config = {
    bucket = var.remote_state_bucket
    key    = "mlops_infra/terraform.tfstate"
    region = var.region
  }
}

data "terraform_remote_state" "rds" {
  backend = "s3"
  config = {
    bucket = var.remote_state_bucket
    key    = "rds/terraform.tfstate"
    region = var.region
  }
}

data "rds_data_source" "rds" {
  rds_instance_id = data.terraform_remote_state.rds.outputs.rds_instance_id
}

# Using 
data "aws_db_instance" "rds_instance" {
  db_instance_identifier = var.rds_instance_id
}

data "aws_secretsmanager_secret" "rds_secret" {
  name = "rds-credentials-${data.aws_db_instance.rds_instance.id}"
}

data "aws_secretsmanager_secret_version" "rds_secret_version" {
  secret_id = data.aws_secretsmanager_secret.rds_secret.id
}

output "rds_endpoint" {
  value = data.aws_db_instance.rds_instance.endpoint
}

output "rds_username" {
  value = data.aws_secretsmanager_secret_version.rds_secret_version.secret_string_json.username
}

output "rds_password" {
  value = data.aws_secretsmanager_secret_version.rds_secret_version.secret_string_json.password
}

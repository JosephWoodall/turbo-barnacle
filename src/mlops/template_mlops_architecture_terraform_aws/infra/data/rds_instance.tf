resource "aws_db_instance" "main" {
  engine           = var.rds_engine
  engine_version   = var.rds_engine_version
  instance_class   = var.rds_instance_class
  allocated_storage = var.rds_allocated_storage
  identifier       = var.rds_instance_identifier
  name             = var.rds_name
  username         = var.rds_username
  password         = var.rds_password
  parameter_group_name = var.rds_parameter_group_name
  skip_final_snapshot = var.rds_skip_final_snapshot
  vpc_security_group_ids = [aws_security_group.rds.id]

  tags = {
    Name = "${var.environment}-rds-${var.rds_instance_identifier}"
    Environment = var.environment
  }
}

data "aws_db_instance" "main" {
  identifier = var.rds_instance_identifier

  depends_on = [aws_db_instance.main]
}

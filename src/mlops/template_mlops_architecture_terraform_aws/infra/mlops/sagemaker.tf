# Create an IAM role that Amazon SageMaker can assume
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "sagemaker-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# Attach a policy to the IAM role that allows Amazon SageMaker to access S3
resource "aws_iam_role_policy_attachment" "sagemaker_s3_access" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  role       = aws_iam_role.sagemaker_execution_role.name
}

# Create a SageMaker endpoint configuration
resource "aws_sagemaker_endpoint_configuration" "example" {
  name_prefix      = "example-endpoint-config"
  production_variants {
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
    model_name             = aws_sagemaker_model.example.name
    variant_name           = "example-variant"
  }
}

# Create a SageMaker endpoint
resource "aws_sagemaker_endpoint" "example" {
  name               = "example-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.example.name
  tags = {
    Name = "Example SageMaker endpoint"
  }
}

# Create a SageMaker model
resource "aws_sagemaker_model" "example" {
  name          = "example-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn
  primary_container {
    image               = "012345678910.dkr.ecr.us-west-2.amazonaws.com/my-model:latest"
    model_data_url      = "s3://my-model-bucket/model.tar.gz"
    environment         = {
      variables = {
        "MY_VARIABLE" = "my-value"
      }
    }
  }
}

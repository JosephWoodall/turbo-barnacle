# Define AWS Lambda layer for Python packages

# Define the layer name and description
resource "aws_lambda_layer_version" "python_packages" {
  filename   = "python_packages.zip"
  layer_name = "python_packages"
  description = "A layer containing commonly used Python packages"

  # Add the Python packages to the layer
  compatible_runtimes = ["python3.7"]
}

# Create a policy to allow access to the layer
resource "aws_iam_policy" "lambda_layer_policy" {
  name        = "lambda-layer-policy"
  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:GetLayerVersion"
        ]
        Resource = aws_lambda_layer_version.python_packages.arn
      }
    ]
  })
}

# Attach the policy to a new IAM role
resource "aws_iam_role" "lambda_layer_role" {
  name = "lambda-layer-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  # Attach the policy to the IAM role
  policy = aws_iam_policy.lambda_layer_policy.arn
}

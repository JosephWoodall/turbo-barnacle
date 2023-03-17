# AWS Lambda Function
resource "aws_lambda_function" "example_lambda_function" {
  function_name    = "example-lambda-function"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.8"
  memory_size      = 128
  timeout          = 10
  filename         = "lambda_function.zip"
  source_code_hash = filebase64sha256("lambda_function.zip")

  environment {
    variables = {
      EXAMPLE_VAR = "example_value"
    }
  }

  tags = {
    Terraform   = "true"
    Environment = var.environment
  }
}

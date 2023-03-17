data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir = "${path.module}/src"

  output_path = "${path.module}/deployment.zip"
}

resource "aws_lambda_function" "predictor" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = var.function_name
  role             = var.lambda_execution_role
  handler          = "handler.handler"
  runtime          = "python3.8"
  memory_size      = var.lambda_memory_size
  timeout          = var.lambda_timeout
  publish          = true

  environment {
    variables = {
      ARTIFACTS_BUCKET = var.artifacts_bucket
      ARTIFACT_PREFIX  = var.artifact_prefix
    }
  }

  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
}

resource "aws_lambda_event_source_mapping" "predictor_event_source_mapping" {
  event_source_arn  = var.event_source_arn
  function_name     = aws_lambda_function.predictor.function_name
  starting_position = "LATEST"
}

resource "aws_lambda_permission" "event_source_permissions" {
  statement_id  = "AllowExecutionFromEventSource"
  action       = "lambda:InvokeFunction"
  function_name = aws_lambda_function.predictor.arn
  principal    = "events.amazonaws.com"

  source_arn = var.event_source_arn
}

data "template_file" "predictor_lambda_role_policy" {
  template = file("${path.module}/templates/predictor-lambda-role-policy.tpl")

  vars = {
    artifacts_bucket = var.artifacts_bucket
    artifact_prefix  = var.artifact_prefix
  }
}

resource "aws_iam_role_policy" "predictor_lambda_role_policy" {
  name = "predictor_lambda_role_policy"
  role = var.lambda_execution_role

  policy = data.template_file.predictor_lambda_role_policy.rendered
}

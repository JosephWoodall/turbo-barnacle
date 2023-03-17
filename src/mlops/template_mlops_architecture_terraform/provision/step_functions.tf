resource "aws_sfn_state_machine" "mlops_pipeline" {
  name     = "mlops_pipeline"
  role_arn = aws_iam_role.mlops_pipeline.arn

  definition = jsonencode({
    Comment = "MLOps pipeline",
    StartAt = "Preprocess Data",
    States = {
      "Preprocess Data" = {
        Type       = "Task",
        Resource   = aws_lambda_function.preprocess_data.arn,
        Next       = "Train Model",
        Catch      = [{
          ErrorEquals = ["States.ALL"],
          Next        = "Cleanup Resources"
        }]
      },
      "Train Model" = {
        Type       = "Task",
        Resource   = aws_lambda_function.train_model.arn,
        Next       = "Evaluate Model",
        Catch      = [{
          ErrorEquals = ["States.ALL"],
          Next        = "Cleanup Resources"
        }]
      },
      "Evaluate Model" = {
        Type       = "Task",
        Resource   = aws_lambda_function.evaluate_model.arn,
        Next       = "Deploy Model",
        Catch      = [{
          ErrorEquals = ["States.ALL"],
          Next        = "Cleanup Resources"
        }]
      },
      "Deploy Model" = {
        Type       = "Task",
        Resource   = aws_lambda_function.deploy_model.arn,
        Next       = "Cleanup Resources",
        Catch      = [{
          ErrorEquals = ["States.ALL"],
          Next        = "Cleanup Resources"
        }]
      },
      "Cleanup Resources" = {
        Type      = "Task",
        Resource  = aws_lambda_function.cleanup_resources.arn,
        End       = true
      }
    }
  })
}

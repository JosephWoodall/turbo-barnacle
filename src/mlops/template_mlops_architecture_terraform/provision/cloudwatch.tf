resource "aws_cloudwatch_metric_alarm" "mlops_lambda_error_alarm" {
  alarm_name          = "mlops_lambda_error_alarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = "1"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "60"
  statistic           = "Sum"
  threshold           = "1"

  dimensions = {
    FunctionName = aws_lambda_function.preprocess_data.function_name
  }

  alarm_description = "This metric monitor Lambda errors for the preprocess_data function"
  alarm_actions     = [aws_sns_topic.mlops_notifications.arn]
}

resource "aws_cloudwatch_metric_alarm" "mlops_lambda_throttle_alarm" {
  alarm_name          = "mlops_lambda_throttle_alarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = "1"
  metric_name         = "Throttles"
  namespace           = "AWS/Lambda"
  period              = "60"
  statistic           = "Sum"
  threshold           = "1"

  dimensions = {
    FunctionName = aws_lambda_function.preprocess_data.function_name
  }

  alarm_description = "This metric monitor Lambda throttles for the preprocess_data function"
  alarm_actions     = [aws_sns_topic.mlops_notifications.arn]
}

output "api_endpoint" {
  value = aws_api_gateway_deployment.mlops-gateway.invoke_url
}

output "predictor_arn" {
  value = aws_lambda_function.predictor.arn
}

output "predictor_logs" {
  value = aws_cloudwatch_log_group.predictor_logs.name
}

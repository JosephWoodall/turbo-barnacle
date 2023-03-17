resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action       = "lambda:InvokeFunction"
  function_name = aws_lambda_function.predictor.arn
  principal    = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_deployment.mlops-gateway.execution_arn}/*/*"
}

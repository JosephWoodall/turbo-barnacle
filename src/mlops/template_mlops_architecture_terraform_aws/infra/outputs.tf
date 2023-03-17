output "vpc_id" {
  value       = aws_vpc.main.id
  description = "The ID of the VPC"
}

output "public_subnet_ids" {
  value       = aws_subnet.public.*.id
  description = "A list of IDs for the public subnets"
}

output "private_subnet_ids" {
  value       = aws_subnet.private.*.id
  description = "A list of IDs for the private subnets"
}

output "nat_gateway_ids" {
  value       = aws_nat_gateway.main.*.id
  description = "A list of IDs for the NAT gateways"
}

output "bastion_instance_ip" {
  value       = aws_instance.bastion.public_ip
  description = "The public IP address of the bastion instance"
}

output "web_server_instance_ips" {
  value       = aws_instance.web.*.private_ip
  description = "A list of private IP addresses for the web server instances"
}

output "rds_endpoint" {
  value       = aws_db_instance.main.endpoint
  description = "The endpoint of the main RDS instance"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.main.id
  description = "The name of the main S3 bucket"
}

output "api_gateway_url" {
  value       = aws_api_gateway_deployment.main.invoke_url
  description = "The URL for the API Gateway deployment"
}

output "lambda_function_arn" {
  value       = aws_lambda_function.main.arn
  description = "The ARN of the main Lambda function"
}

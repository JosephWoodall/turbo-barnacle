variable "project_prefix" {
  type        = string
  description = "A prefix to add to all resources created by this module"
}

variable "region" {
  type        = string
  default     = "us-east-1"
  description = "The AWS region in which to create resources"
}

variable "vpc_cidr" {
  type        = string
  default     = "10.0.0.0/16"
  description = "The CIDR block for the VPC"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
  description = "A list of CIDR blocks for the public subnets"
}

variable "private_subnet_cidrs" {
  type        = list(string)
  default     = ["10.0.3.0/24", "10.0.4.0/24"]
  description = "A list of CIDR blocks for the private subnets"
}

variable "ami_id" {
  type        = string
  default     = "ami-0c55b159cbfafe1f0"
  description = "The ID of the Amazon Machine Image (AMI) to use for the EC2 instances"
}

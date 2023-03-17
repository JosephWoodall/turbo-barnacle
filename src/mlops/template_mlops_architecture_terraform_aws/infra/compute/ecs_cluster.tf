# Define the ECS cluster
resource "aws_ecs_cluster" "main" {
  name = var.ecs_cluster_name

  tags = {
    Terraform   = "true"
    Environment = var.environment
  }
}

resource "aws_ecr_repository" "ml_ops_ecr_repo" {
  name = var.ecr_repo_name

  image_tag_mutability = "MUTABLE"

  tags = {
    Name = "ml-ops-ecr-repository"
  }
}

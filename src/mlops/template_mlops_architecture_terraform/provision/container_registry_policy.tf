resource "aws_ecr_lifecycle_policy" "ml_ops_ecr_lifecycle_policy" {
  repository = aws_ecr_repository.ml_ops_ecr_repo.name
  policy     = jsonencode({
    rules = [
      {
        rulePriority = 1,
        description  = "Expire images older than 30 days",
        selection    = {
          tagStatus = "untagged",
          countType = "sinceImagePushed",
          countUnit = "days",
          countNumber = 30
        },
        action = {
          type = "expire"
        }
      }
    ]
  })
}

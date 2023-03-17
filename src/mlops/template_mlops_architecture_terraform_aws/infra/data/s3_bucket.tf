resource "aws_s3_bucket" "main" {
  bucket = var.s3_bucket_name

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  lifecycle {
    prevent_destroy = false
  }
}

output "s3_bucket_id" {
  value       = aws_s3_bucket.main.id
  description = "The ID of the S3 bucket"
}

output "s3_bucket_arn" {
  value       = aws_s3_bucket.main.arn
  description = "The ARN of the S3 bucket"
}

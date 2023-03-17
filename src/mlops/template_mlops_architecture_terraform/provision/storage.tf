resource "aws_s3_bucket" "ml_ops_s3_bucket" {
  bucket = var.s3_bucket_name

  versioning {
    enabled = true
  }

  tags = {
    Name = "ml-ops-s3-bucket"
  }
}

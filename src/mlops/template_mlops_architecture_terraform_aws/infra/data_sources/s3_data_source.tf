locals {
  s3_bucket_name = var.s3_bucket_name
}

data "aws_s3_bucket" "s3_bucket" {
  bucket = local.s3_bucket_name

  dynamic "tags" {
    for_each = var.s3_bucket_tags
    content {
      key   = tags.key
      value = tags.value
    }
  }
}

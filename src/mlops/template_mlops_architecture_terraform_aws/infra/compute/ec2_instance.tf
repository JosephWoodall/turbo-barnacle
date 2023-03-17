resource "aws_instance" "ml_instance" {
  ami           = var.ec2_ami_id
  instance_type = var.instance_type
  subnet_id     = var.private_subnet_id
  key_name      = var.key_name
  vpc_security_group_ids = [
    aws_security_group.ingress_sg.id,
    aws_security_group.egress_sg.id
  ]
  user_data = base64encode(local.script)

  root_block_device {
    volume_size = var.volume_size
  }

  tags = {
    Name = "ml_instance"
  }
}

data "template_file" "user_data" {
  template = file("${path.module}/scripts/user_data.sh.tpl")

  vars = {
    s3_bucket = var.s3_bucket
    region    = var.region
  }
}

locals {
  script = data.template_file.user_data.rendered
}

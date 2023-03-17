resource "aws_vpc" "ml_ops_vpc" {
  cidr_block = var.vpc_cidr

  tags = {
    Name = "ml-ops-vpc"
  }
}

resource "aws_subnet" "ml_ops_subnet" {
  cidr_block = var.subnet_cidr
  vpc_id     = aws_vpc.ml_ops_vpc.id

  tags = {
    Name = "ml-ops-subnet"
  }
}

resource "aws_security_group" "ml_ops_sg" {
  name_prefix = "ml-ops-sg"

  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ml-ops-security-group"
  }
}

# Create a security group for the VPC
resource "aws_security_group" "vpc" {
  name_prefix = "mlops-vpc-sg-"

  # Allow all inbound traffic within the VPC
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "all"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }

  # Allow outbound traffic to the Internet
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "all"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Create a security group for the Bastion instance
resource "aws_security_group" "bastion" {
  name_prefix = "mlops-bastion-sg-"
  vpc_id      = aws_vpc.main.id

  # Allow inbound SSH traffic from the Internet
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow outbound traffic to the Internet
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "all"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Create a security group for the web servers
resource "aws_security_group" "web" {
  name_prefix = "mlops-web-sg-"
  vpc_id      = aws_vpc.main.id

  # Allow inbound HTTP and HTTPS traffic from the Internet
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow outbound traffic to the Internet
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "all"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow inbound traffic from the Bastion instance
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    security_groups = [aws_security_group.bastion.id]
  }
}

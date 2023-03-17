resource "aws_instance" "ml_ops_ec2_instance" {
  ami           = var.ec2_ami_id
  instance_type = var.ec2_instance_type
  subnet_id     = aws_subnet.ml_ops_subnet.id
  vpc_security_group_ids = [
    aws_security_group.ml_ops_sg.id,
  ]

  user_data = <<-EOF
              #!/bin/bash
              echo "Hello World" > /tmp/hello.txt
              EOF

  tags = {
    Name = "ml-ops-ec2-instance"
  }
}

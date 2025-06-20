aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 475080169237.dkr.ecr.ap-south-1.amazonaws.com/dev/text2sql

docker pull 475080169237.dkr.ecr.ap-south-1.amazonaws.com/dev/text2sql:latest

docker run -d 475080169237.dkr.ecr.ap-south-1.amazonaws.com/dev/text2sql:latest
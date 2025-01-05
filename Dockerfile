FROM ubuntu:latest

ARG AWS_ACCESS_KEY_ID

ARG AWS_SECRET_ACCESS_KEY

ARG AWS_REGION

RUN apt update -y && apt install awscli -y

RUN aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID \
&& aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY \
&& aws configure set region $AWS_REGION

RUN aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin \
654654307335.dkr.ecr.ap-south-1.amazonaws.com

FROM 654654307335.dkr.ecr.ap-south-1.amazonaws.com/dev/text2sql_python_dependency:latest
#FROM text2sql_python_dependency:latest

#FROM python:3.10-slim

EXPOSE 8000

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y

RUN pip install --upgrade pip && pip install -r requirements1.txt

CMD ["chainlit","run", "app.py"]
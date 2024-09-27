FROM ubuntu:latest

FROM python:3.10-slim

ARG AWS_ACCESS_KEY_ID

ARG AWS_SECRET_ACCESS_KEY

ARG AWS_ECR_REGION

EXPOSE 8000

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["chainlit","run", "app.py"]
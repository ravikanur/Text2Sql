FROM ubuntu:latest

ARG ECR_REGISTRY

ARG ECR_REPOSITORY

#FROM python:3.10-slim

FROM $ECR_REGISTRY/$ECR_REPOSITORY:latest

EXPOSE 8000

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y

RUN pip install --upgrade pip && pip install -r requirements1.txt

CMD ["chainlit","run", "app.py"]
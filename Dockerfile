FROM ubuntu:latest

FROM python:3.10-slim

COPY . /app

WORKDIR /app

EXPOSE 8000

RUN apt-get update && apt-get install -y

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["chainlit","run", "app.py"]
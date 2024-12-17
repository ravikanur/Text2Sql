FROM ubuntu:latest

FROM text2sql_python_dependency

#FROM python:3.10-slim

EXPOSE 8000

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y

RUN pip install --upgrade pip && pip install -r requirements1.txt

CMD ["chainlit","run", "app.py"]
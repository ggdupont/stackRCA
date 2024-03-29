FROM python:3.8-slim-buster

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY *.py ./

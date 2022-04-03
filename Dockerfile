FROM python:3.9-slim

RUN apk add --no-cache libpq-devel

# create and setup working directory
RUN mkdir /app

# install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-input -r /app/requirements.txt

# install code
COPY main.py /app/main.py

WORKDIR /app

ENTRYPOINT python main.py 2>&1
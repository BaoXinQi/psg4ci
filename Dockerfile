FROM python:3.10-slim-bookworm

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt

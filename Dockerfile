FROM jupyter/minimal-notebook:x86_64-python-3.10.11

USER root

# added new line to get lsb-release
RUN apt-get update && apt-get install -y lsb-release && apt-get clean all && \
    apt-get install -y gnupg && \
    apt-get install -y curl && \
    apt-get install -y libpq-dev && \
    apt-get install -y python3-dev && \
    apt-get install -y gcc

WORKDIR /python-docker

# RUN apt update && \ 
#     apt install -y python3-venv python3-pip && \
#     pip install --upgrade pip 

RUN python3.10 -m ensurepip 
RUN python3.10 -m pip install --upgrade pip

COPY requirements.txt ./
RUN python3.10 -m pip install -r requirements.txt

# Install  codebase
RUN mkdir -p /python-docker/src
COPY src/ /python-docker/src
RUN python3.10 -m pip install -e src

COPY .env ./
#COPY app.py ./



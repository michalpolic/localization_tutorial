FROM colmap/colmap:latest
LABEL maintainer="Paul-Edouard Sarlin"

ARG PYTHON_VERSION=3.8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y python3-distutils
    
RUN apt-get update -y && \
    apt-get install -y unzip wget software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

RUN apt-get install -y git

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install notebook && \
    pip install ipywidgets --upgrade

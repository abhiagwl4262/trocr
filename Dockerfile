ARG BASE_IMAGE="nvcr.io/nvidia/pytorch:21.02-py3"
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update 
RUN apt install -y libgl1-mesa-glx 
RUN apt-get install -y libmagickwand-dev

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

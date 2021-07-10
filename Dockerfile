# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

# RUN sudo python3 -m pip install -U pip
# RUN sudo python3 -m pip install -U setuptools


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         libgcc-5-dev \
         ca-certificates \

    && rm -rf /var/lib/apt/lists/*


RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

#RUN apt-get -y update && apt-get install -y --no-install-recommends \
#         wget \
#         curl \
#         python3.6 \
#         nginx \
#		 libgcc-5-dev \
#         ca-certificates \
#         python3-distutils \
#    && rm -rf /var/lib/apt/lists/*

#RUN python3.6 -m pip install -U pip

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.


#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.6 get-pip.py && \
#    pip3 install numpy==1.19.2 scipy scikit-learn==0.24.1   pandas==1.1.5 flask gevent gunicorn && \
##        (cd /usr/local/lib/python3.6/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
##        rm -rf /root/.cache

RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip

RUN pip3  install --default-timeout=1000 boto3~=1.17.58 pillow~=8.1.2 torch~=1.8.0 torchvision~=0.9.0 argparse~=1.4.0 pandas~=1.1.5 thop~=0.0.31-2005241907 tqdm~=4.59.0 numpy~=1.19.5 opencv-python~=4.5.1.48 matplotlib~=3.3.4 s3fs~=2021.4.0 scipy~=1.5.4
# --no-cache-dir
# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY xgboost /opt/program
WORKDIR /opt/program


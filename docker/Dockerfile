
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

MAINTAINER karavaev-sasha@mail.ru

RUN mkdir /car

# Install simulator
RUN apt-get update && apt-get install -y wget unzip && cd /car && wget "https://github.com/tawnkramer/gym-donkeycar/releases/download/v2020.5.16/DonkeySimLinux.zip" \
    && unzip DonkeySimLinux

COPY gym-donkeycar /car/gym-donkeycar
COPY rc_car /car/rc_car
COPY requirements.txt /car/

# OpenCV and other deps
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install deps for Unity 
RUN apt-get update \
    && apt install -y cmake \
                   libopenmpi-dev \
                   libglu1-mesa \
                   libgl1-mesa-glx \
                   freeglut3 \
                   freeglut3-dev \
                   swig \
    && rm -rf /var/lib/apt/lists/*

RUN cd /car && pip install -r requirements.txt 
RUN ls /car/
RUN cd /car/gym-donkeycar && pip install -r requirements_dev.txt && python setup.py install
COPY setup.py /car/
RUN cd /car/ && python setup.py install


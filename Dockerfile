FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Install base dependencies
RUN \
    apt-get -y update \
    && apt-get -y install \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        wget

# Add cmake repository
RUN \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

# Install library dependencies
RUN \
    apt-get -y update \
    && apt-get -y install \
        cmake \
        g++ \
        gcc \
        libboost-dev \
        libopenmpi-dev

# Build the library
CMD \
    cd /energymanager \
    && rm -rf ./build \
    && mkdir ./build \
    && cd ./build \
    && cmake --configure ../ \
    && cmake --build ../ --target all \
    && tail -f /dev/null
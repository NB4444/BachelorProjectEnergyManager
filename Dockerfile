FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Install base dependencies
RUN \
    apt-get -y update \
    && apt-get -y install \
        apt-transport-https \
        ca-certificates \
        git \
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
        g++-8 \
        gcc-8 \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-math-dev \
        libboost-program-options-dev \
        libboost-serialization-dev \
        libboost-stacktrace-dev \
        libboost-test-dev \
        libopenmpi-dev \
        libsqlite3-dev

# Install samples
RUN \
    mkdir -p /applications \
    && cd /applications \
    && git clone --depth 1 --branch v10.1 https://github.com/NVIDIA/cuda-samples.git \
    && cd ./cuda-samples \
    && make

# Compile the library
CMD \
    cd /EnergyManager \
    && mkdir -p ./Resources \
    && mkdir -p ./cmake-build-docker \
    && cd ./cmake-build-docker \
    && cmake .. \
    && cmake --build . --target all \
    && echo The container is running! Use ./InteractiveShell.sh to connect and execute operations. \
    && tail -f /dev/null

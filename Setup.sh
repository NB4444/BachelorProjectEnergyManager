#!/bin/bash

# Uninstall old versions
sudo apt-get -y remove \
  containerd \
  docker \
  docker-compose \
  docker-engine \
  docker.io \
  nvidia-container-runtime \
  runc

# Install dependencies
sudo apt-get -y update
sudo apt-get -y install \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release

# Add the docker repository GPG keys
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --yes --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

# Install docker
sudo apt-get -y update
sudo apt-get -y install \
  containerd.io \
  docker-ce \
  docker-ce-cli

# Retrieve the distribution
#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
distribution="ubuntu20.04"

# Add the nvidia container runtime repository GPG keys
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -

# Add the nvidia container runtime repository
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

# Install the nvidia container runtime
sudo apt-get -y update
sudo apt-get -y install nvidia-container-runtime

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Restart docker to apply all packages
sudo systemctl restart docker

# Build the image
sudo docker-compose build

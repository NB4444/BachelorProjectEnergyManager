#!/bin/bash

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
buildDirectory="$projectDirectory/cmake-build-default"
resourcesDirectory="$projectDirectory/Resources"

# Generate the build directory
mkdir "$buildDirectory"
cd "$buildDirectory"

# Generate the CMake configuration
cmake "$projectDirectory"

# Build the project
make

# Build the CUDA samples
cd "$resourcesDirectory/CUDA/Samples"
make

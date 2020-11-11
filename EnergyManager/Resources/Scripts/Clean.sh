#!/bin/bash

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
buildDirectory="$projectDirectory/cmake-build-default"
resourcesDirectory="$projectDirectory/Resources"
database="$resourcesDirectory/Test Results/database.sqlite"

rm -r "$buildDirectory"
rm "$database"

# Clean CUDA samples
cd "$resourcesDirectory/CUDA/Samples"
make clean

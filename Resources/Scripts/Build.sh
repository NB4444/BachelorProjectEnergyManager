#!/bin/bash

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
energyManagerDirectory="$projectDirectory/EnergyManager"
energyManagerResourcesDirectory="$energyManagerDirectory/Resources"
visualizerDirectory="$projectDirectory/Visualizer"

"$energyManagerResourcesDirectory/Scripts/Build.sh"

#!/bin/bash

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
resourcesDirectory="$projectDirectory/Resources"
sourceDirectory="$projectDirectory/Source"
executable="$sourceDirectory/Visualizer.py"
interpreter="$projectDirectory/venv/bin/python3"
database="$projectDirectory/../EnergyManager/Resources/Test Results/database.sqlite"

cd "$scriptDirectory"
./Build.sh

"$interpreter" "$executable" --database "$database" --output-directory "$resourcesDirectory/Visualizations"

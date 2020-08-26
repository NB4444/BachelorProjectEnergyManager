#!/bin/bash

scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
projectDirectory="$scriptDirectory/../.."
resourcesDirectory="$projectDirectory/Resources"

rm -r "$resourcesDirectory/Visualizations"

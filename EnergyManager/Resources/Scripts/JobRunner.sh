#!/bin/bash

script="$1"
scriptDirectory="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

cd "$scriptDirectory"
./Build.sh

sbatch "./Job.sh \"$scriptDirectory/$script\""

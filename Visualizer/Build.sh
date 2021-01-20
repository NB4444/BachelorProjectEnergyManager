#!/bin/bash

"$1" env create --prefix "$2/CondaEnvironment" --file "$3/environment.yml"
"$2/CondaEnvironment/bin/jupyter" labextension install @jupyter-widgets/jupyterlab-manager
"$2/CondaEnvironment/bin/jupyter" labextension install @jupyterlab/toc
"$2/CondaEnvironment/bin/jupyter" labextension install jupyter-matplotlib

#!/bin/bash

conda env create --prefix "$1/CondaEnvironment" --file "$2/environment.yml"
"$1/CondaEnvironment/bin/jupyter" labextension install @jupyter-widgets/jupyterlab-manager
"$1/CondaEnvironment/bin/jupyter" labextension install @jupyterlab/toc
"$1/CondaEnvironment/bin/jupyter" labextension install jupyter-matplotlib

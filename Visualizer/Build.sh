#!/bin/bash

"$1" env create --prefix "$2" --file "$3/environment.yml"
"$2/bin/jupyter" labextension install @jupyter-widgets/jupyterlab-manager
"$2/bin/jupyter" labextension install @jupyterlab/toc
"$2/bin/jupyter" labextension install jupyter-matplotlib

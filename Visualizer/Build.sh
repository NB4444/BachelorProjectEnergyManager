#!/bin/bash

"$1" env create --prefix "$2" --file "$3/environment.yml"
"$2/bin/jupyter" labextension install @jupyter-widgets/jupyterlab-manager @jupyterlab/toc jupyter-matplotlib

#!/bin/bash

#
# Please edit CUPTI_DIR
#

export CUPTI_DIR=/hpc/base/cuda/cuda-10.1/extras/CUPTI
export LD_LIBRARY_PATH=${CUPTI_DIR}/lib64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUPTI_DIR}/lib64:${LIBRARY_PATH}
export C_INCLUDE_PATH=${CUPTI_DIR}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${CUPTI_DIR}/include:${CPLUS_INCLUDE_PATH}

# Using SLURM
srun --export="LD_PRELOAD=./../src/libmonitor.so" application

#
# You will have to edit CUDA_ROOT and CUPTO_ROOT directories
#

#CUDA_ROOT  = /hpc/base/cuda/cuda-10.1
CUDA_ROOT  = /usr/local/cuda-10.1
CUPTI_ROOT = $(CUDA_ROOT)/extras/CUPTI

#
# Maybe you don't need to edit this
#

CC         = gcc
CFLAGS     = -Wall -I.
CUDA_CF    = -I$(CUDA_ROOT)/include
CUDA_LD    = -L$(CUDA_ROOT)/lib64 -lcudart -L$(CUDA_ROOT)/lib64/stubs/ -lcuda
CUPTI_CF   = -I$(CUPTI_ROOT)/include
CUPTI_LD   = -L$(CUPTI_ROOT)/lib64 -lcupti
NVML_LD    = -L$(CUDA_ROOT)/lib64/stubs/ -lnvidia-ml

#
# Rules
#

all: clean _common _metrics monitor printer

_common:
	$(CC) $(CFLAGS) -fPIC -c common/msr.c
	$(CC) $(CFLAGS) -fPIC -c common/time.c
	$(CC) $(CFLAGS) -fPIC -c common/pipes.c
	$(CC) $(CFLAGS) -fPIC -c common/topology.c
	$(CC) $(CFLAGS) -fPIC -c common/topology_cpufreq.c
	$(CC) $(CFLAGS) -fPIC -c common/string_enhanced.c

_metrics:
	$(CC) $(CFLAGS) -c energy_gpu.c
	$(CC) $(CFLAGS) $(CUDA_CF) -c metrics/energy_gpu_nvml.c
	$(CC) $(CFLAGS) -c metrics/frequency_intel63.c

monitor:
	$(CC) $(CFLAGS) -c monitor.c
	$(CC) $(CFLAGS) -c apis_suscriptor.c
	$(CC) $(CFLAGS) -o monitor *.o $(NVML_LD) -lpthread

printer:
	$(CC) $(CFLAGS) $(CUDA_CF) $(CUPTI_CF) -fPIC -c libmonitor.c
	$(CC) $(CFLAGS) -shared -o libmonitor.so libmonitor.o time.o pipes.o string_enhanced.o $(CUPTI_LD) -ldl

clean:
	rm -f *.o monitor *.so *.a

fclean: clean
	rm -f core.* gpus.pipe

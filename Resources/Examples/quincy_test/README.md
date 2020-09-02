Functional tests and test tools list
------------------------------------
Functional tests are programs which tests concrete library and daemon functionalities against your system hardware, kernel and libraries.

You will need privileges to execute some of these tests. In case you are running sudo, be sure to append your libraries path to `LD_LIBRARY_PATH` as the following example:

```
sudo LD_LIBRARY_PATH=<freeipmi_lib_path> tests/functionals/freeipmi_overhead
```

Another and more complex example is the SIMD test. It uses *FreeIPMI* and *PAPI*. This is a multithreading test so, binding is recommended. It needs some arguments:

```
Usage: simd_power_metrics n_sockets n_threads n_iterations csv frequency
- n_sockets: number of sockets in the node
- n_threads: threads to create and bind
- n_iterations: number of n_iterations to gather energy metrics
- csv: print output in csv format (0,1)
- frequency: the fixed CPU clock frequency
```

Shell script example:

```
#!/bin/bash

SOCKETS=2
CORES=28
ITERATIONS=100000000
CSV=1
FREQ=2600000

sudo LD_LIBRARY_PATH=freeipmi_lib_path:papi_lib_path OMP_NUM_THREADS=$CORES KMP_AFFINITY=granularity=fine,compact,1,0 ./simd_power_metrics_openmp $SOCKETS $CORES $ITERATIONS $CSV $FREQ
```

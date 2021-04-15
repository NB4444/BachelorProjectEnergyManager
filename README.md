# EnergyManager

This repository contains the files for the collaboration project between EAR and SARA on power saving methods for GPU
kernels. EnergyManager is a framework that can be used to test and monitor heterogeneous CPU-GPU applications in order
to automatically apply energy saving strategies.

## Installation

There is some installation required.
You should perform the following commands in your terminal in the root directory:

```shell script
./Setup.sh
```

This script will install all the necessary dependencies and prepare your system to use docker and the appropriate NVIDIA extensions.
The script was made for Ubuntu 20.04 but can most likely be used with newer versions.

## Usage

To use the framework, first build and start the docker containers using this command:

```shell script
./Run.sh
```

This command will launch a docker container for both the library itself and the Visualizer.
You can press CTRL + C at any point to stop the framework from running if necessary.
Whenever the framework is started using this command, new changes to the source code are detected and if necessary the container will automatically recompile.
Hence, to apply new edits, stop any running containers and use this command to build and re-start the framework.

### Interacting With the Framework

While the framework is running, you can open another terminal and use the following command to interact with the framework:

```shell script
./InteractiveShell.sh
```

This command opens a terminal inside of the docker container where the library has already been pre-compiled and all the necessary dependencies are installed.
You can find the compiled EnergyManager files inside the container at `/EnergyManager/cmake-build-docker`.
Similarly, the root directory of this project, meaning the directory that contains this `README.md` file, will be accessible from inside the container at `/EnergyManager`.

In this environment you can install additional packages and software to test with the framework.
Changes you make inside this environment do not affect your host system, and thus this provides isolation between the framework and the rest of your system.

A sample application can be found inside the container at `/EnergyManager/cmake-build-docker/Example/EnergyManager-Example`.
This application simply runs a profiling session of matrix multiply included in the CUDA samples.
The source code for this application can be found at `/EnergyManager/Example` and outlines how you can implement your own profilers.
Simply copy the Example directory and rename it to whatever you want your new profiler to be named.
Once renamed, add the corresponding `add_subdirectory` command at the end of `CMakeLists.txt` pointing to the new directory.
Whenever you re-start the framework the new profiler will be included and compiled, and the compiled version stored at `/EnergyManager/cmake-build-docker` inside the container.

### Using the Visualizer

While the framework is running, you can visit [http://localhost:8888](http://localhost:8888) to access the Visualizer.
The Visualizer contains notebooks to visualize the data generated during profiling sessions with the EnergyManager library.
Before running the notebooks be sure to have generated some data first!
You can run the example application `EnergyManager/cmake-build-docker/Example/EnergyManager-Example` to generate some sample data.
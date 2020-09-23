# EnergyManager

EnergyManager is a framework that can be used to test and monitor heterogenous CPU-GPU applications in order to automatically apply energy saving strategies.

## Usage

This section describes the different ways in which the EnergyManager framework can be used.

### Scripts

Some sample scripts can be found in the `Resources/Scripts` directory.

### Command Line

EnergyManager can be used by providing arguments when starting the executable.
The available arguments are as follows:

| Argument    | Shorthand | Format     | Description                                     |
| :---------- | :-------- | :--------- | :---------------------------------------------- |
| --test      | -t        | NAME       | The name of the test to run.                    |
| --parameter | -p        | NAME=VALUE | A parameter to provide to the test.             |
| --database  | -d        | FILE       | The database file to store the test results in. |

#### Available Tests

There are several tests available, each with their own parameters.
They are as follows:

| Test                             | Parameters                                                                                                                                                   |
| :------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FixedFrequencyMatrixMultiplyTest | name, cpu, gpu, minimumCPUFrequency, maximumCPUFrequency, minimumGPUFrequency, maximumGPUFrequency, matrixAWidth, matrixAHeight, matrixBWidth, matrixBHeight |
| MatrixMultiplyTest               | name, cpu, gpu, matrixAWidth, matrixAHeight, matrixBWidth, matrixBHeight                                                                                     |
| PingTest                         | name, host, times                                                                                                                                            |
| SyntheticGPUWorkloadTest*        | name, workload, cpu, gpu
| VectorAddSubtractTest            | name, gpu, computeCount                                                                                                                                      |

* The `SyntheticGPUWorkloadTest` supports multiple workload types, each with their own parameters.
See the table below.

| Workload               | Parameters                                               |
| :--------------------- | :------------------------------------------------------- |
| ActiveInactiveWorkload | activeOperations, inactivePeriod, cycles                 |
| AllocateFreeWorkload   | hostAllocations, hostSize, deviceAllocations, deviceSize |
| VectorAddWorkload      | size                                                     |

## Compiling

This section describes the process required to compile EnergyManager locally.

### Dependencies

To compile EnergyManager locally, you will need to have a few dependencies installed on your system.
These are as follows:

- CMake 3.12.1
- SQLite3
- CUDA 10.1

#### Debian

They can be installed using the following command on Debian based distributions (assuming the correct repositories are available on the system):

```shell script
apt-get install -y \
    cmake \
    sqlite3 \
    cuda-10-1
```

### Running CMake

To compile the project, simply run the following command wherever you wish to place the build files:

```shell script
cmake ${PATH_TO_PROJECT_ROOT}
```
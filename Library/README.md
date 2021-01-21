# Library

The Library contains all core functionality provided by the EnergyManager.

## Usage

This section outlinescommon use cases for the Library along with detailed instructions.

### Profiling an Application

The most common use case for the Library is profiling applications.
In this scenario, the Library will take care of launching the desired application and monitoring its resource consumption as well as other behavioural characteristics.

To get started, first you need to create a new class that inherits the `EnergyManager::Profiling::Profilers::Profiler` class.
Use the following template to get started profiling a simple application:
```c++
#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Profiling/Profilers/Profiler.hpp>
#include <EnergyManager/Utility/Application.hpp>

using EnergyManager::Hardware::CPU;
using EnergyManager::Hardware::GPU;
using EnergyManager::Profiling::Profilers::Profiler;
using EnergyManager::Utility::Application;

class MyProfiler : public Profiler {
protected:
    void onProfile(const std::map<std::string, std::string>& profile) final {
        Application("/bin/ping", std::vector<std::string> { "8.8.8.8" }, { CPU::getCPU(0) }, GPU::getGPU(0), true, true, true).run();
    }
};
```

This will set up a profiler that starts the ping application with the argument 8.8.8.8 on CPU 0 and GPU 0.
# Library

The Library contains all core functionality provided by the EnergyManager.

## Usage

This section outlines common use cases for the Library along with detailed instructions.

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
#include <EnergyManager/Utility/Text.hpp>

using EnergyManager::Hardware::CPU;
using EnergyManager::Hardware::GPU;
using EnergyManager::Profiling::Profilers::Profiler;
using EnergyManager::Utility::Application;
using EnergyManager::Utility::Text;

int main(int argumentCount, char* argumentValues[]) {
    static auto arguments = Text::parseArgumentsMap(argumentCount, argumentValues);
    
    class MyProfiler : public Profiler {
    protected:
        void onProfile(const std::map<std::string, std::string>& profile) final {
            Application(
                // The path to the application to launch
                "/bin/ping",
                
                // The parameters to pass to the application
                std::vector<std::string> {
                    "8.8.8.8"
                },
                
                // The CPUs to use to run the application
                {
                    CPU::getCPU(0)
                },
                
                // The GPU to use to run the application
                GPU::getGPU(0),
                
                // Whether to log application output
                true,
                
                // Whether to inject the library reporter into the application which enables some additional metrics to be measured
                true
            ).run();
        }
        
    public:
        MyProfiler() : Profiler(
            // The name of the profiler
            "MyProfiler",
            
            // The profiles
            {
                {
                    { "core", "0" },
                    { "gpu", "0" }
                }
            },
            
            // Other arguments that can be specified on the command line
            arguments
        ) {
        }
    };
    
    MyProfiler().run();
    
    return 0;
}
```

When an instance of this object is created, 
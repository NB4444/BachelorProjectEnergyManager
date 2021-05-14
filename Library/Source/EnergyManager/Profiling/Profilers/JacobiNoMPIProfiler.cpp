#include "./JacobiNoMPIProfiler.hpp"

#include <EnergyManager.hpp>

void JacobiNoMPIProfiler::onProfile(const std::map<std::string, std::string>& profile) {
	// The CPU to use to run the application
	unsigned int cpuID = 0;
	
	// The GPU to use to run the application
	unsigned int gpuID = 0;
	
	auto path = std::string(JACOBI_DIRECTORY) + "/jacobi";
	EnergyManager::Utility::Logging::logInformation("Application path: %s", path.c_str());
	
	EnergyManager::Utility::Application(
		// The path to the application to launch, in this case we will be launching Jacobi
		path,
		
		// The parameters to pass to the application
		// We extract these values from the current profile
		std::vector<std::string> {
			"--file",
			profile.at("file"),
			"-i",
			profile.at("Ni"),
			"-j",
			profile.at("Nj"),
			"-n",
			profile.at("iterations"),
			"-k",
			profile.at("kernel"),
		},
		
		// Pass the CPUs to use to run the application
		//{ EnergyManager::Hardware::CPU::getCPU(cpuID) },
		// Core to use to run the application
		{ EnergyManager::Hardware::Core::getCore(cpuID) },
		
		// Pass the GPU to use to run the application
		EnergyManager::Hardware::GPU::getGPU(gpuID),
		
		// Whether to log application output
		true,
		
		// Whether to inject the library reporter into the application which enables some additional metrics to be measured
		// We set it to true here but it is only really effective for CUDA applications
		true)
		.run();
}

JacobiNoMPIProfiler::JacobiNoMPIProfiler(const std::map<std::string, std::string>& arguments)
	: Profiler(
	// The name of the profiler
	"JacobiNoMPIProfiler",
	
	// The profiles
	{
		//{ {"file", "/applications/jacobi/small"}, {"Ni", "512"}, {"Nj", "512"}, {"iterations", "10000"}, {"kernel", "1"}},
		//
		//{ {"file", "/applications/jacobi/small"}, {"Ni", "512"}, {"Nj", "512"}, {"iterations", "10000"}, {"kernel", "2"} },
		
		{ {"file", "/applications/jacobi/medium"}, {"Ni", "1024"}, {"Nj", "1024"}, {"iterations", "10000"}, {"kernel", "1"}},

		{ {"file", "/applications/jacobi/medium"}, {"Ni", "1024"}, {"Nj", "1024"}, {"iterations", "10000"}, {"kernel", "2"} },
		
		//{ {"file", "/applications/jacobi/large"}, {"Ni", "2048"}, {"Nj", "2048"}, {"iterations", "10000"}, {"kernel", "1"}},
		//
		//{ {"file", "/applications/jacobi/large"}, {"Ni", "2048"}, {"Nj", "2048"}, {"iterations", "10000"}, {"kernel", "2"} },
	},
	arguments) {
	setIterationsPerRun(3);
	setRunsPerProfile(1);
	setRandomize(false);
}


#include "./StreamclusterProfiler.hpp"

#include <EnergyManager.hpp>

void LavaMDProfiler::onProfile(const std::map<std::string, std::string>& profile) {
	// The CPU to use to run the application
	unsigned int cpuID = 0;
	
	// The GPU to use to run the application
	unsigned int gpuID = 0;
	
	auto path = std::string(RODINIA_DIRECTORY) + "/cuda/lavaMD/lavaMD";
	EnergyManager::Utility::Logging::logInformation("Application path: %s", path.c_str());
	
	EnergyManager::Utility::Application(
		// The path to the application to launch, in this case we will be launching streamcluster
		path,
		
		// The parameters to pass to the application
		// We extract these values from the current profile
		std::vector<std::string> { "-boxes1d", profile.at("boxes1d") },
		
		// Pass the CPUs to use to run the application
		{ EnergyManager::Hardware::CPU::getCPU(cpuID) },
		
		// Pass the GPU to use to run the application
		EnergyManager::Hardware::GPU::getGPU(gpuID),
		
		// Whether to log application output
		true,
		
		// Whether to inject the library reporter into the application which enables some additional metrics to be measured
		// We set it to true here but it is only really effective for CUDA applications
		true)
		.run();
}

LavaMDProfiler::LavaMDProfiler(const std::map<std::string, std::string>& arguments)
	: Profiler(
	// The name of the profiler
	"LavaMDProfiler",
	
	// The profiles
	{
		{ {"boxes1d", "10"} },
	},
	arguments) {
	setIterationsPerRun(3);
	setRunsPerProfile(1);
	setRandomize(false);
}


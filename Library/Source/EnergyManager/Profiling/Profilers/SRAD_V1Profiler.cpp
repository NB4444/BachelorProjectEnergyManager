#include "./SRAD_V1Profiler.hpp"

#include <EnergyManager.hpp>

void SRAD_V1Profiler::onProfile(const std::map<std::string, std::string>& profile) {
	// The CPU to use to run the application
	unsigned int cpuID = 0;
	
	// The GPU to use to run the application
	unsigned int gpuID = 0;
	
	auto path = std::string(RODINIA_DIRECTORY) + "/cuda/srad/srad_v1/srad";
	EnergyManager::Utility::Logging::logInformation("Application path: %s", path.c_str());
	
	EnergyManager::Utility::Application(
		// The path to the application to launch, in this case we will be launching SRAD_V1
		path,
		
		// The parameters to pass to the application
		// We extract these values from the current profile
		std::vector<std::string> { profile.at("iterations"), profile.at("coefficient"), profile.at("rows"), profile.at("columns") },
		
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

SRAD_V1Profiler::SRAD_V1Profiler(const std::map<std::string, std::string>& arguments)
	: Profiler(
	// The name of the profiler
	"SRAD_V1Profiler",
	
	// The profiles
	{
		{ {"iterations", "100"}, {"coefficient", "0.5"}, {"rows", "502"}, {"columns", "458"} },
	},
	arguments) {
	setIterationsPerRun(3);
	setRunsPerProfile(1);
	setRandomize(false);
}


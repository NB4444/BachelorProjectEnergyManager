#include "./StreamclusterProfiler.hpp"

#include <EnergyManager.hpp>

void StreamclusterProfiler::onProfile(const std::map<std::string, std::string>& profile) {
	// The CPU to use to run the application
	unsigned int cpuID = 0;

	// The GPU to use to run the application
	unsigned int gpuID = 0;

	auto path = std::string(RODINIA_DIRECTORY) + "/cuda/streamcluster/sc_gpu";
	EnergyManager::Utility::Logging::logInformation("Application path: %s", path.c_str());

	EnergyManager::Utility::Application(
		// The path to the application to launch, in this case we will be launching streamcluster
		std::string(RODINIA_DIRECTORY) + "/cuda/streamcluster/sc_gpu",

		// The parameters to pass to the application
		// We extract these values from the current profile
		std::vector<std::string> { profile.at("k1"), profile.at("k2"), profile.at("d"), profile.at("n"),
								             profile.at("chunksize"), profile.at("clustersize"), profile.at("infile"),
								   			 profile.at("outfile"), profile.at("nproc")},

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

StreamclusterProfiler::StreamclusterProfiler(const std::map<std::string, std::string>& arguments)
	: Profiler(
		// The name of the profiler
		"StreamclusterProfiler",

		// The profiles
		{
			{ {"k1", "10"}, {"k2", "20"}, {"d", "256"}, {"n", "65536"},
			  {"chunksize", "65536"}, {"clustersize", "1000"}, {"infile", "none"},
			  {"outfile", "output.txt"}, {"nproc", "1"}  },
		},

		// We can forward the command line arguments here
		arguments) {
	// We can configure the amount of iterations in a single profiling session - this means that within one single profiling session of one of the provided profiles, the onProfile function will be called this amount of times
	// Setting this to a value higher than 1 can be useful if you want to investigate multiple executions of the same application, or if you want to warm up the cache
	setIterationsPerRun(3);

	// We can also configure how often to repeat a profiling session
	// This means that every profile of the provided profiles will be profiled this amount of times
	// Setting this to a value higher than 1 can be useful to ensure the repeatability of an experiment
	setRunsPerProfile(1);

	// You can determine whether to randomize the profiling order of the provided profiles
	// This may be useful to prevent all short tests from being at the start and long tests at the end or vice versa, which can cause the framework to provide inaccurate time estimations
	// It is set to true by default
	// We can set it to false here since we only run a few short tests
	setRandomize(false);
}

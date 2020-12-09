#include <EnergyManager/Monitoring/Monitors/EnergyMonitor.hpp>
#include <EnergyManager/Profiling/Profilers/BFSProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUBLASProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUFFTProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/JacobiProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/KMeansProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/MatrixMultiplyProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>
#include <chrono>

const auto energySavingInterval = std::chrono::milliseconds(10);

void matrixMultiplyControlShort(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setIterationsPerRun(3);
	profiler.setRunsPerProfile(50);

	profiler.run();
}

void matrixMultiplyControlMedium(const std::map<std::string, std::string>& arguments) {
	const auto core = EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
	const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	std::vector<std::map<std::string, std::string>> profiles = profiler.getProfiles();
	for(auto& profile : profiles) {
		profile["minimumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
		profile["maximumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
		profile["minimumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
		profile["maximumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	}
	profiler.setProfiles(profiles);
	profiler.setIterationsPerRun(15);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void matrixMultiplyFixedFrequenciesShort(std::map<std::string, std::string> arguments) {
	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "8";
	arguments["--gpuCoreClockRatesToProfile"] = "8";

	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setIterationsPerRun(3);
	profiler.setRunsPerProfile(1);

	profiler.run();
}

void matrixMultiplyEnergyMonitorMedium(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor)");
	profiler.setIterationsPerRun(15);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true));

	profiler.run();
}

void kMeansControlShort(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	profiler.setIterationsPerRun(15);
	profiler.setRunsPerProfile(50);

	profiler.run();
}

void kMeansControlMedium(const std::map<std::string, std::string>& arguments) {
	const auto core = EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
	const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	std::vector<std::map<std::string, std::string>> profiles = profiler.getProfiles();
	for(auto& profile : profiles) {
		profile["minimumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
		profile["maximumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
		profile["minimumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
		profile["maximumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	}
	profiler.setProfiles(profiles);
	profiler.setIterationsPerRun(75);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void kMeansFixedFrequenciesShort(std::map<std::string, std::string> arguments) {
	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "8";
	arguments["--gpuCoreClockRatesToProfile"] = "8";

	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	profiler.setIterationsPerRun(15);
	profiler.setRunsPerProfile(1);

	profiler.run();
}

void kMeansEnergyMonitorMedium(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor)");
	profiler.setIterationsPerRun(75);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true));

	profiler.run();
}

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	//std::make_shared<EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler>(arguments);
	//std::make_shared<EnergyManager::Profiling::Profilers::BFSProfiler>(arguments);
	//std::make_shared<EnergyManager::Profiling::Profilers::CUBLASProfiler>(arguments);
	//std::make_shared<EnergyManager::Profiling::Profilers::CUFFTProfiler>(arguments);
	//std::make_shared<EnergyManager::Profiling::Profilers::JacobiProfiler>(arguments);

	// Generate the control data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyControlShort(arguments));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansControlShort(arguments));

	// Test fixed frequencies
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyFixedFrequenciesShort(arguments));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansFixedFrequenciesShort(arguments));

	// Test energy monitor
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyControlMedium(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansControlMedium(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyEnergyMonitorMedium(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansEnergyMonitorMedium(arguments));

	return 0;
}
#include <EnergyManager/Monitoring/Monitors/EnergyMonitor.hpp>
#include <EnergyManager/Profiling/Profilers/BFSProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUBLASProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUFFTProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/JacobiProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/KMeansProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/MatrixMultiplyProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>

void test(std::map<std::string, std::string>& arguments) {
	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "2";
	arguments["--gpuCoreClockRatesToProfile"] = "2";

	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor)");
	profiler.setIterationsPerRun(1);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		std::chrono::milliseconds(25)));

	profiler.run();
}

void test2(std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor)");
	profiler.setIterationsPerRun(3);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		std::chrono::milliseconds(25),
		true));

	profiler.run();
}

void test3(std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	profiler.setIterationsPerRun(15);
	profiler.setRunsPerProfile(1);

	profiler.run();
}

void matrixMultiplyControl(std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setIterationsPerRun(3);
	profiler.setRunsPerProfile(30);

	profiler.run();
}

void matrixMultiplyFixedFrequencies(std::map<std::string, std::string>& arguments) {
	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "8";
	arguments["--gpuCoreClockRatesToProfile"] = "8";

	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setIterationsPerRun(3);
	profiler.setRunsPerProfile(1);

	profiler.run();
}

void matrixMultiplyEnergyMonitor(std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor)");
	profiler.setIterationsPerRun(3);
	profiler.setRunsPerProfile(30);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		std::chrono::milliseconds(25),
		true));

	profiler.run();
}

void kMeansControl(std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	profiler.setIterationsPerRun(5);
	profiler.setRunsPerProfile(30);

	profiler.run();
}

void kMeansFixedFrequencies(std::map<std::string, std::string>& arguments) {
	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "8";
	arguments["--gpuCoreClockRatesToProfile"] = "8";

	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	profiler.setIterationsPerRun(5);
	profiler.setRunsPerProfile(1);

	profiler.run();
}

void kMeansEnergyMonitor(std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor)");
	profiler.setIterationsPerRun(5);
	profiler.setRunsPerProfile(30);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		std::chrono::milliseconds(25),
		true));

	profiler.run();
}

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	//std::make_shared<EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler>(arguments);
	//std::make_shared<EnergyManager::Profiling::Profilers::BFSProfiler>(arguments);
	//std::make_shared<EnergyManager::Profiling::Profilers::CUBLASProfiler>(arguments);
	//std::make_shared<EnergyManager::Profiling::Profilers::CUFFTProfiler>(arguments);
	//std::make_shared<EnergyManager::Profiling::Profilers::JacobiProfiler>(arguments);

	//test3(arguments);
	//return 0;

	// Generate the control data
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyControl(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansControl(arguments));

	// Test fixed frequencies
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyFixedFrequencies(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansFixedFrequencies(arguments));

	// Test energy monitor
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyEnergyMonitor(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansEnergyMonitor(arguments));

	return 0;
}
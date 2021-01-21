#pragma once

#include "Configuration.hpp"

#include <EnergyManager/Monitoring/Monitors/EnergyMonitor.hpp>
#include <EnergyManager/Profiling/Profilers/BFSProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUBLASProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/CUFFTProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/JacobiProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/KMeansProfiler.hpp>
#include <EnergyManager/Profiling/Profilers/MatrixMultiplyProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>
#include <chrono>

const unsigned int matrixMultiplyShortIterations = 3;
const unsigned int matrixMultiplyMediumIterations = 15;

void matrixMultiplyControlShort(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setIterationsPerRun(matrixMultiplyShortIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void matrixMultiplyControlMedium(const std::map<std::string, std::string>& arguments) {
	const auto core = EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
	const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	//std::vector<std::map<std::string, std::string>> profiles = profiler.getProfiles();
	//for(auto& profile : profiles) {
	//	profile["minimumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
	//	profile["maximumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
	//	profile["minimumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	//	profile["maximumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	//}
	//profiler.setProfiles(profiles);
	profiler.setIterationsPerRun(matrixMultiplyMediumIterations);
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
	arguments["--cpuCoreClockRatesToProfile"] = "10";
	arguments["--gpuCoreClockRatesToProfile"] = "10";

	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setIterationsPerRun(matrixMultiplyShortIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void matrixMultiplyEnergyMonitorMediumSimple(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Simple)");
	profiler.setIterationsPerRun(matrixMultiplyMediumIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true,
		false));

	profiler.run();
}

void matrixMultiplyEnergyMonitorMediumSmart(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Smart)");
	profiler.setIterationsPerRun(matrixMultiplyMediumIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true,
		true));

	profiler.run();
}

void matrixMultiply(const std::map<std::string, std::string>& arguments) {
	// Control data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyControlShort(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyControlMedium(arguments));

	// Fixed frequency data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyFixedFrequenciesShort(arguments));

	// Energy monitor data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyEnergyMonitorMediumSimple(arguments));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyEnergyMonitorMediumSmart(arguments));
}
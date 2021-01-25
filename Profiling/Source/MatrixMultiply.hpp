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

void matrixMultiplyControl(const std::map<std::string, std::string>& arguments, const unsigned int& iterations) {
	EnergyManager::Utility::Logging::logInformation("Profiling Matrix Multiply control (%d iterations)...", iterations);

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
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void matrixMultiplyFixedFrequencies(std::map<std::string, std::string> arguments, const unsigned int& iterations) {
	EnergyManager::Utility::Logging::logInformation("Profiling Matrix Multiply fixed frequencies (%d iterations)...", iterations);

	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "3";
	arguments["--gpuCoreClockRatesToProfile"] = "3";

	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void matrixMultiplyEnergyMonitor(const std::map<std::string, std::string>& arguments, const unsigned int& iterations, const bool& smart) {
	EnergyManager::Utility::Logging::logInformation("Profiling Matrix Multiply energy monitor (%d iterations, smart %d)...", iterations, smart);

	auto profiler = EnergyManager::Profiling::Profilers::MatrixMultiplyProfiler(arguments);
	if(smart) {
		profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Smart)");
	} else {
		profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Simple)");
	}
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true,
		smart));

	profiler.run();
}

void matrixMultiply(const std::map<std::string, std::string>& arguments) {
	const unsigned int shortIterations = 3;
	const unsigned int mediumIterations = 15;

	// Control data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyControl(arguments, mediumIterations));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyControl(arguments, shortIterations));

	// Fixed frequency data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyFixedFrequencies(arguments, mediumIterations));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyFixedFrequencies(arguments, shortIterations));

	// Energy monitor data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyEnergyMonitor(arguments, mediumIterations, false));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyEnergyMonitor(arguments, mediumIterations, true));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyEnergyMonitor(arguments, shortIterations, false));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(matrixMultiplyEnergyMonitor(arguments, shortIterations, true));
}
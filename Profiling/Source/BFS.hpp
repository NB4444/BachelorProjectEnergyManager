#pragma once

#include "Configuration.hpp"

#include <EnergyManager/Monitoring/Monitors/EnergyMonitor.hpp>
#include <EnergyManager/Profiling/Profilers/BFSProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>
#include <chrono>

void bfsControl(const std::map<std::string, std::string>& arguments, const unsigned int& iterations) {
	EnergyManager::Utility::Logging::logInformation("Profiling Jacobi control (%d iterations)...", iterations);

	const auto core = EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
	const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

	auto profiler = EnergyManager::Profiling::Profilers::BFSProfiler(arguments);
	std::vector<std::map<std::string, std::string>> profiles = profiler.getProfiles();
	for(auto& profile : profiles) {
		profile["minimumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
		profile["maximumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
		profile["minimumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
		profile["maximumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	}
	profiler.setProfiles(profiles);
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(5);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void bfsFixedFrequencies(std::map<std::string, std::string> arguments, const unsigned int& iterations) {
	EnergyManager::Utility::Logging::logInformation("Profiling Jacobi fixed frequencies (%d iterations)...", iterations);

	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "7";
	arguments["--gpuCoreClockRatesToProfile"] = "7";

	auto profiler = EnergyManager::Profiling::Profilers::BFSProfiler(arguments);
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void bfsEnergyMonitor(const std::map<std::string, std::string>& arguments, const unsigned int& iterations, const bool& system) {
	EnergyManager::Utility::Logging::logInformation("Profiling Jacobi energy monitor (%d iterations, smart %d)...", iterations, system);

	auto profiler = EnergyManager::Profiling::Profilers::BFSProfiler(arguments);
	if(system) {
		profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor System)");
	} else {
		profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor MinMax)");
	}
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(2);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true,
		system));

	profiler.run();
}

void bfs(const std::map<std::string, std::string>& arguments) {
	const unsigned int shortIterations = 15;
	const unsigned int mediumIterations = 75;

	// Control data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(bfsControl(arguments, shortIterations));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(bfsControl(arguments, mediumIterations));
	
	// Energy monitor data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(bfsEnergyMonitor(arguments, shortIterations, false));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(bfsEnergyMonitor(arguments, shortIterations, true));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(bfsEnergyMonitor(arguments, mediumIterations, false));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(bfsEnergyMonitor(arguments, mediumIterations, true));

	// Fixed frequency data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(bfsFixedFrequencies(arguments, shortIterations));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(bfsFixedFrequencies(arguments, mediumIterations));
}
#pragma once

#include "Configuration.hpp"

#include <EnergyManager/Monitoring/Monitors/EnergyMonitor.hpp>
#include <EnergyManager/Profiling/Profilers/KMeansProfiler.hpp>
#include <EnergyManager/Utility/Text.hpp>
#include <chrono>

void kMeansControl(const std::map<std::string, std::string>& arguments, const unsigned int& iterations) {
	EnergyManager::Utility::Logging::logInformation("Profiling K-Means control (%d iterations)...", iterations);

	const auto core = EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
	const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	//std::vector<std::map<std::string, std::string>> profiles = profiler.getProfiles();
	//for(auto& profile : profiles) {
	//	profile["minimumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
	//	profile["maximumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
	//	profile["minimumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	//	profile["maximumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	//}
	//profiler.setProfiles(profiles);
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false,
		halfingPeriod,
		doublingPeriod));

	profiler.run();
}

void kMeansFixedFrequencies(std::map<std::string, std::string> arguments, const unsigned int& iterations) {
	EnergyManager::Utility::Logging::logInformation("Profiling K-Means fixed frequencies (%d iterations)...", iterations);

	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "7";
	arguments["--gpuCoreClockRatesToProfile"] = "7";

	const auto core = EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
	//const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	std::vector<std::map<std::string, std::string>> profiles = profiler.getProfiles();
	for(auto& profile : profiles) {
		//profile["minimumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMinimumCoreClockRate());
		profile["maximumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
		//profile["minimumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMinimumCoreClockRate());
		//profile["maximumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	}
	profiler.setProfiles(profiles);
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false,
		halfingPeriod,
		doublingPeriod));

	profiler.run();
}

void kMeansEnergyMonitor(const std::map<std::string, std::string>& arguments, const unsigned int& iterations, const bool& system) {
	EnergyManager::Utility::Logging::logInformation("Profiling K-Means energy monitor (%d iterations, smart %d)...", iterations, system);

	const auto core = EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
	//const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

	auto profiler = EnergyManager::Profiling::Profilers::KMeansProfiler(arguments);
	if(system) {
		profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor System)");
	} else {
		profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor MinMax)");
	}
	std::vector<std::map<std::string, std::string>> profiles = profiler.getProfiles();
	for(auto& profile : profiles) {
		//profile["minimumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMinimumCoreClockRate());
		profile["maximumCPUClockRate"] = EnergyManager::Utility::Text::toString(core->getMaximumCoreClockRate());
		//profile["minimumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMinimumCoreClockRate());
		//profile["maximumGPUClockRate"] = EnergyManager::Utility::Text::toString(gpu->getMaximumCoreClockRate());
	}
	profiler.setProfiles(profiles);
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true,
		halfingPeriod,
		doublingPeriod,
		system));

	profiler.run();
}

void kMeans(const std::map<std::string, std::string>& arguments) {
	const unsigned int shortIterations = 15;
	const unsigned int mediumIterations = 75;

	// Control data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansControl(arguments, shortIterations));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansControl(arguments, mediumIterations));

	// Energy monitor data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansEnergyMonitor(arguments, shortIterations, false));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansEnergyMonitor(arguments, shortIterations, true));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansEnergyMonitor(arguments, mediumIterations, false));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansEnergyMonitor(arguments, mediumIterations, true));

	// Fixed frequency data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansFixedFrequencies(arguments, shortIterations));
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(kMeansFixedFrequencies(arguments, mediumIterations));
}
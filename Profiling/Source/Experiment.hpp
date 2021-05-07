#pragma once

#include "Configuration.hpp"

#include <EnergyManager.hpp>
#include <chrono>

template <class T>
void experimentControl(const std::map<std::string, std::string>& arguments,  const unsigned int& iterations) {
	auto profiler = T(arguments);
	EnergyManager::Utility::Logging::logInformation("Profiling " + profiler.getProfileName() + " control (%d iterations)...", iterations);
	profiler.setIterationsPerRun(iterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		//EnergyManager::Hardware::CPU::getCPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--cpu", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false,
		halfingPeriod,
		doublingPeriod));

	profiler.run();
}

template <class T>
void experimentEnergyMonitor(const std::map<std::string, std::string>& arguments, const unsigned int& iterations, const enum Policies& system) {
	auto profiler = T(arguments);
	
	EnergyManager::Utility::Logging::logInformation("Profiling " + profiler.getProfileName() + " energy monitor (%d iterations, smart %d)...", iterations, system);
	
	if(system) {
		profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor System)");
	} else {
		profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor MinMax)");
	}
	std::vector<std::map<std::string, std::string>> profiles = profiler.getProfiles();
	
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

template <class T>
void experiment(const std::map<std::string, std::string>& arguments, unsigned int iterations) {
	auto i_max = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "-i", 1);
	auto policies = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--policies", 0);

	for(int i = 0; i < i_max; ++i) {
		// Control data
		ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentControl<T>(arguments, iterations));

		if(policies == 0) {
			// Energy monitor data
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::Minmax));
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::System));
		}
	}
}
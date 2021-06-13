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
		doublingPeriod,
		true));
	
	profiler.run();
}

template <class T>
void experimentEnergyMonitor(const std::map<std::string, std::string>& arguments, const unsigned int& iterations, const enum Policies& policy, const bool singleCore = true, const int percentage = 100) {
	auto profiler = T(arguments);
	
	EnergyManager::Utility::Logging::logInformation("Profiling " + profiler.getProfileName() + " energy monitor (%d iterations, smart %d)...", iterations, system);
	
	switch(policy) {
		case Policies::System:
			profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor System)");
			break;
		case Policies::Minmax:
			profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor MinMax)");
			break;
		case Policies::RankedMinmax:
			profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Ranked MinMax)");
			break;
		case Policies::ScalingMinmax:
			profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Scaled MinMax)");
			break;
		case Policies::MaxFreq:
			profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Max frequency)");
			break;
		case Policies::StaticFreq:
			profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Static frequency " + std::to_string(percentage) + "%)");
			break;
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
		singleCore,
		percentage,
		policy));
	
	profiler.run();
}

template <class T>
void experiment(const std::map<std::string, std::string>& arguments, unsigned int iterations) {
	auto i_max = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "-i", 1);
	bool multicore = EnergyManager::Utility::Text::getArgument<bool>(arguments, "--multicore", false);
	auto no_policies = EnergyManager::Utility::Text::getArgument<bool>(arguments, "--nopolicies", true);
	auto staticFreq = EnergyManager::Utility::Text::getArgument<bool>(arguments, "--staticFreq", false);
	auto singleCore = !multicore;
	
	for(int i = 0; i < i_max; ++i) {
		// Control data
		ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentControl<T>(arguments, iterations));
		if (!staticFreq) {
			if(no_policies) {
				// Energy monitor data
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::Minmax, singleCore));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::System, singleCore));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::MaxFreq, singleCore));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::RankedMinmax, singleCore));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::ScalingMinmax, singleCore));
			}
		} else {
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::StaticFreq, singleCore, 10));
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::StaticFreq, singleCore, 25));
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::StaticFreq, singleCore, 50));
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::StaticFreq, singleCore, 75));
			//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(experimentEnergyMonitor<T>(arguments, iterations, Policies::StaticFreq, singleCore, 100));
		}
	}
}

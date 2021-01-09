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

const unsigned int jacobiShortIterations = 15;
const unsigned int jacobiMediumIterations = 75;

void jacobiControlShort(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::JacobiProfiler(arguments);
	profiler.setIterationsPerRun(jacobiShortIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void jacobiControlMedium(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::JacobiProfiler(arguments);
	profiler.setIterationsPerRun(jacobiMediumIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void jacobiFixedFrequenciesShort(std::map<std::string, std::string> arguments) {
	arguments["--fixedClockRates"] = "1";
	arguments["--cpuCoreClockRatesToProfile"] = "10";
	arguments["--gpuCoreClockRatesToProfile"] = "10";

	auto profiler = EnergyManager::Profiling::Profilers::JacobiProfiler(arguments);
	profiler.setIterationsPerRun(jacobiShortIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		false));

	profiler.run();
}

void jacobiEnergyMonitorMediumSimple(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::JacobiProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Simple)");
	profiler.setIterationsPerRun(jacobiMediumIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true,
		false));

	profiler.run();
}

void jacobiEnergyMonitorMediumSmart(const std::map<std::string, std::string>& arguments) {
	auto profiler = EnergyManager::Profiling::Profilers::JacobiProfiler(arguments);
	profiler.setProfileName(profiler.getProfileName() + " (EnergyMonitor Smart)");
	profiler.setIterationsPerRun(jacobiMediumIterations);
	profiler.setRunsPerProfile(1);
	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true,
		true));

	profiler.run();
}

void jacobi(const std::map<std::string, std::string>& arguments) {
	// Control data
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(jacobiControlShort(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(jacobiControlMedium(arguments));

	// Fixed frequency data
	//ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(jacobiFixedFrequenciesShort(arguments));

	// Energy monitor data
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(jacobiEnergyMonitorMediumSimple(arguments));
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(jacobiEnergyMonitorMediumSmart(arguments));
}
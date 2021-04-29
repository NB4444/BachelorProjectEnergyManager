#pragma once

#include <EnergyManager.hpp>
#include <memory>

class StreamclusterProfiler : public EnergyManager::Profiling::Profilers::Profiler {
	using EnergyManager::Profiling::Profilers::Profiler::Profiler;

protected:
	/**
	 * Gets called whenever a profiler session is initiated and is provided with the current profile to run.
	 * @param profile The profile to run, consisting of named variables and their associated values.
	 */
	void onProfile(const std::map<std::string, std::string>& profile) final;

public:
	/**
	 * Creates a new StreamclusterProfiler.
	 * @param arguments The command line arguments that were passed when starting.
	 */
	explicit StreamclusterProfiler(const std::map<std::string, std::string>& arguments);
};
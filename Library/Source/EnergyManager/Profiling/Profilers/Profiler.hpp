#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/Core.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Hardware/Processor.hpp"
#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Profiling/Persistence/ProfilerSession.hpp"
#include "EnergyManager/Utility/Runnable.hpp"
#include "EnergyManager/Utility/Units/Hertz.hpp"

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			/**
			 * Runs an operation and profiles its execution.
			 */
			class Profiler : public Utility::Runnable {
				/**
				 * The name of the profile.
				 */
				std::string profileName_;

				/**
				 * The monitors to run during runtime.
				 */
				std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors_;

				/**
				 * The profiles to test.
				 */
				std::vector<std::map<std::string, std::string>> profiles_;

				/**
				 * The amount of runs to perform per profile.
				 */
				unsigned int runsPerProfile_;

				/**
				 * The amount of iterations to do in a single run without restarting the Monitors.
				 */
				unsigned int iterationsPerRun_;

				/**
				 * The amount of times to re-try a run before halting.
				 */
				unsigned int retriesPerRun_;

				/**
				 * The last profiling sessions.
				 */
				std::vector<std::shared_ptr<Persistence::ProfilerSession>> profilerSessions_;

				/**
				 * Whether to randomize the order the profiles are evaluated in.
				 */
				bool randomize_;

				/**
				 * Whether to automatically save profiler sessions.
				 */
				bool autoSave_;

				/**
				 * Whether to use SLURM.
				 */
				bool slurm_;

				/**
				 * Whether to use EAR.
				 */
				bool ear_;

				/**
				 * The interval at which to monitor EAR.
				 */
				std::chrono::system_clock::duration earMonitorInterval_;

				/**
				 * The arguments to use for SLURM.
				 */
				std::map<std::string, std::string> slurmArguments_;

				/**
				 * Does one profiling run.
				 * @param profile The profile.
				 * @param attempts The amount of attempts to make.
				 */
				void runProfile(const std::map<std::string, std::string>& profile, const unsigned int& attempts = 3);

				/**
				 * Does one profiling run using SLURM.
				 * @param profile The profile.
				 * @param attempts The amount of attempts to make.
				 */
				void runSLURMProfile(const std::map<std::string, std::string>& profile, const unsigned int& attempts = 3);

			protected:
				std::vector<std::string> generateHeaders() const override;

				void beforeRun() final;

				void onRun() final;

				/**
				 * Before profiling a specific workload.
				 * @param profile The profile to use.
				 */
				virtual void beforeProfile(const std::map<std::string, std::string>& profile);

				/**
				 * Profiles the workload.
				 * @param profile The profile to use.
				 */
				virtual void onProfile(const std::map<std::string, std::string>& profile);

				/**
				 * After profiling a specific workload.
				 * @param profile The profile to use.
				 */
				virtual void afterProfile(const std::map<std::string, std::string>& profile, const std::shared_ptr<Persistence::ProfilerSession>& profilerSession);

			public:
				/**
				 * Generates a bunch of intervals between a minimum and maximum value.
				 * @param minimumValue The minimum value.
				 * @param maximumValue The maximum value.
				 * @param intervals The amount of values to pick between the minimm and maximum (inclusive).
				 * @return The values.
				 */
				template<typename Type>
				static std::vector<Type> generateValueRange(const Type& minimumValue, const Type& maximumValue, const unsigned int& intervals) {
					const auto coreClockRateIntervalSize = intervals > 1 ? (maximumValue - minimumValue) / (intervals - 1) : 0;

					std::vector<Type> values;
					for(unsigned int clockRateIndex = 0; clockRateIndex < intervals; ++clockRateIndex) {
						values.emplace_back(minimumValue + clockRateIndex * coreClockRateIntervalSize);
					}

					return values;
				}

				/**
				 * Generates a bunch of profiles for each of the available processor frequencies specified.
				 * @param minimumClockRate The minimum clock rate to profile.
				 * @param maximumClockRate The maximum clock rate to profile.
				 * @param clockRatesToProfile The amount of profiles to test in between the minimum and maximum clock rate (inclusive).
				 * @return The frequency profiles.
				 */
				static std::vector<Utility::Units::Hertz>
					generateFrequencyValueRange(const Utility::Units::Hertz& minimumClockRate, const Utility::Units::Hertz& maximumClockRate, const unsigned int& clockRatesToProfile);

				/**
				 * Generates a bunch of profiles for each of the available processor frequencies specified.
				 * @param processor The processor to use.
				 * @param clockRatesToProfile The amount of profiles to test in between the minimum and maximum clock rate (inclusive).
				 * @return The frequency profiles.
				 */
				static std::vector<Utility::Units::Hertz> generateFrequencyValueRange(const std::shared_ptr<Hardware::Processor>& processor, const unsigned int& clockRatesToProfile);

				/**
				 * Generates a copy of the provided profiles for every frequency combination.
				 * @param profiles The profiles to use as input.
				 * @param core The core to use.
				 * @param coreClockRatesToProfile The amount of core clock rates to profile.
				 * @param gpu The GPU to use.
				 * @param gpuClockRatesToProfile The amount of GPU clock rates to profile.
				 * @return The fixed frequency profiles.
				 */
				static std::vector<std::map<std::string, std::string>> generateFixedFrequencyProfiles(
					const std::vector<std::map<std::string, std::string>>& profiles,
					const std::shared_ptr<Hardware::Core>& core,
					const unsigned int& coreClockRatesToProfile,
					const std::shared_ptr<Hardware::GPU>& gpu,
					const unsigned int& gpuClockRatesToProfile);

				/**
				 * Creates a new Profiler.
				 * @param profileName The name of the profile.
				 * @param profiles The profiles to test.
				 * @param monitors The monitors to use when running.
				 * @param runsPerProfile The amount of runs per profile.
				 * @param iterationsPerRun The amount of iterations to do in a single run without restarting the Monitors.
				 * @param retriesPerRun The amount of times to attempt to re-try a run before halting.
				 * @param randomize Whether to randomize profile evaluation order.
				 * @param autoSave Whether to automatically save profiler sessions.
				 * @param slurm Whether to use SLURM.
				 * @param slurmArguments The arguments to use for SLURM.
				 * @param ear Whether to use EAR.
				 * @param earMonitorInterval The interval at which to run the EAR monitor.
				 */
				explicit Profiler(
					std::string profileName,
					std::vector<std::map<std::string, std::string>> profiles,
					std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors,
					const unsigned int& runsPerProfile = 1,
					const unsigned int& iterationsPerRun = 1,
					const unsigned int& retriesPerRun = 10,
					const bool& randomize = false,
					const bool& autoSave = false,
					const bool& slurm = false,
					const std::map<std::string, std::string>& slurmArguments = {},
					const bool& ear = false,
					const std::chrono::system_clock::duration& earMonitorInterval = std::chrono::system_clock::duration(0));

				/**
				 * Creates a new Profiler from command line arguments.
				 * @param profileName The name of the profile.
				 * @param profiles The profiles to test.
				 * @param arguments The command line arguments.
				 */
				explicit Profiler(const std::string& profileName, const std::vector<std::map<std::string, std::string>>& profiles, const std::map<std::string, std::string>& arguments);

				/**
				 * Gets the name of the profile.
				 * @return The profile name.
				 */
				std::string getProfileName() const;

				/**
				 * Sets the name of the profile.
				 * @param profileName The profile name.
				 */
				void setProfileName(const std::string& profileName);

				/**
				 * Gets the profiles to test.
				 * @return The profiles.
				 */
				std::vector<std::map<std::string, std::string>> getProfiles() const;

				/**
				 * Sets the profiles to test.
				 * @param profiles The profiles.
				 */
				void setProfiles(const std::vector<std::map<std::string, std::string>>& profiles);

				/**
				 * Gets the amount of runs to perform per profile.
				 * @return The amount of runs.
				 */
				unsigned int getRunsPerProfile() const;

				/**
				 * Sets the amount of runs to perform per profile.
				 * @param runsPerProfile The amount of runs.
				 */
				void setRunsPerProfile(const unsigned int& runsPerProfile);

				/**
				 * Gets the amount of iterations per run.
				 * @return The amount of iterations per run.
				 */
				unsigned int getIterationsPerRun() const;

				/**
				 * Sets the amount of iterations per run.
				 * @param iterationsPerRun The amount of iterations per run.
				 */
				void setIterationsPerRun(const unsigned int& iterationsPerRun);

				/**
				 * Gets the last profiling session.
				 * @return The last profiling session.
				 */
				std::vector<std::shared_ptr<Persistence::ProfilerSession>> getProfilerSessions() const;

				/**
				 * Gets whether to randomize profile evaluation order.
				 * @return Whether to randomize profile evaluation order.
				 */
				bool getRandomize() const;

				/**
				 * Sets whether to randomize profile evaluation order.
				 * @param randomize Whether to randomize profile evaluation order.
				 */
				void setRandomize(const bool& randomize);

				/**
				 * Gets whether to automatically save the profiler sessions as they are generated.
				 * @return Whether to auto save the profiler sessions.
				 */
				bool getAutoSave() const;

				/**
				 * Sets whether to automatically save the profiler sessions as they are generated.
				 * @param autoSave Whether to auto save the profiler sessions.
				 */
				void setAutoSave(const bool& autoSave);
			};
		}
	}
}
#pragma once

#include "EnergyManager/Monitoring/Monitors/Monitor.hpp"
#include "EnergyManager/Monitoring/Persistence/MonitorData.hpp"
#include "EnergyManager/Monitoring/Persistence/ProfilerSession.hpp"
#include "EnergyManager/Utility/Runnable.hpp"

namespace EnergyManager {
	namespace Monitoring {
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
			 * The last profiling sessions.
			 */
			std::vector<std::shared_ptr<Persistence::ProfilerSession>> profilerSessions_;

			/**
			 * Whether to randomize the order the profiles are evaluated in.
			 */
			bool randomize_;

		protected:
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
			 * Creates a new Profiler.
			 * @param profileName The name of the profile.
			 * @param profiles The profiles to test.
			 * @param monitors The monitors to use when running.
			 * @param runsPerProfile The amount of runs per profile.
			 * @param iterationsPerRun The amount of iterations to do in a single run without restarting the Monitors.
			 * @param randomize Whether to randomize profile evaluation order.
			 */
			explicit Profiler(
				std::string profileName,
				std::vector<std::map<std::string, std::string>> profiles,
				std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors,
				const unsigned int& runsPerProfile = 1,
				const unsigned int& iterationsPerRun = 1,
				const bool& randomize = false);

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
		};
	}
}
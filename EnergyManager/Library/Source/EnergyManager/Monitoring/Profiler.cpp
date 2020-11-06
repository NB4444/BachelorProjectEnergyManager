#include "./Profiler.hpp"

#include "EnergyManager/Utility/Logging.hpp"

#include <algorithm>
#include <random>
#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		void Profiler::beforeRun() {
			profilerSessions_.clear();
		}

		void Profiler::onRun() {
			// Copy the profiles
			std::vector<std::map<std::string, std::string>> profiles = getProfiles();

			// Shuffle if necessary
			if(getRandomize()) {
				auto random = std::default_random_engine { static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count()) };
				std::shuffle(profiles.begin(), profiles.end(), random);
			}

			// Process all profiles
			for(unsigned int profileIndex = 0; profileIndex < getProfiles().size(); ++profileIndex) {
				const auto& profile = profiles[profileIndex];

				for(unsigned int runIndex = 0; runIndex < getRunsPerProfile(); ++runIndex) {
					try {
						Utility::Logging::logInformation("Preparing profiler execution...");
						beforeProfile(profile);

						EnergyManager::Utility::Logging::logInformation(
							"Profiling profile %d/%d, run %d/%d, with profile {%s}...",
							profileIndex + 1,
							getProfiles().size(),
							runIndex + 1,
							getRunsPerProfile(),
							Utility::Text::join(profile, ", ", " = ").c_str());
						if(!(profileIndex == 0 && runIndex == 0)) {
							const auto timeSinceStart = std::chrono::system_clock::now() - getStartTimestamp();
							const auto completedRuns = profileIndex * getRunsPerProfile() + runIndex;
							const auto timePerRun = timeSinceStart / completedRuns;
							const auto runsLeft = getProfiles().size() * getRunsPerProfile() - completedRuns;
							const auto timeToEnd = runsLeft * timePerRun;

							EnergyManager::Utility::Logging::logInformation("Time left: %s", Utility::Text::formatDuration(timeToEnd).c_str());
						}

						// Start the monitor threads
						for(auto& monitor : monitors_) {
							Utility::Logging::logInformation("Starting monitor %s thread...", monitor->getName().c_str());
							monitor->reset();
							monitor->run(true);
						}

						// Profile the workload
						Utility::Logging::logInformation("Profiling workload...");
						for(unsigned int iterationIndex = 0; iterationIndex < getIterationsPerRun(); ++iterationIndex) {
							onProfile(profile);
						}

						// Stop all monitors threads
						for(auto& monitor : monitors_) {
							Utility::Logging::logInformation("Stopping monitor %s...", monitor->getName().c_str());
							monitor->stop();
						}

						// Wait for the monitors to finish
						for(auto& monitor : monitors_) {
							monitor->synchronize();
						}

						//// Pretty-print the monitor results
						//for(const auto& monitor : monitors_) {
						//	Utility::Logging::logInformation("Monitor %s results:", monitor->getName().c_str());
						//	for(const auto& timestampedValue : monitor->getVariableValues()) {
						//		auto timestamp = timestampedValue.first;
						//		auto variables = timestampedValue.second;
						//
						//		for(const auto& variableValues : variables) {
						//			auto name = variableValues.first;
						//			auto value = variableValues.second;
						//			auto timestampString = Utility::Text::formatTimestamp(timestamp);
						//
						//			Utility::Logging::logInformation("\t[%s] %s = %s", timestampString.c_str(), name.c_str(), value.c_str());
						//		}
						//	}
						//}

						Utility::Logging::logInformation("Collecting profiler session data...");

						// Set up the session
						auto profilerSession = std::make_shared<Persistence::ProfilerSession>(getProfileName(), profile);

						// Get the monitor data and add it to the session
						std::vector<std::shared_ptr<Persistence::MonitorData>> monitorData;
						for(auto& monitor : monitors_) {
							monitorData.push_back(std::make_shared<Persistence::MonitorData>(monitor->getVariableValues(), monitor->getName(), profilerSession));
						}
						profilerSession->setMonitorData(monitorData);

						Utility::Logging::logInformation("Finalizing profiler execution...");
						afterProfile(profile, profilerSession);

						// Add the session to the collection
						profilerSessions_.push_back(profilerSession);
					} catch(const Utility::Exceptions::Exception& exception) {
						exception.log();
					}
				}
			}
		}

		void Profiler::beforeProfile(const std::map<std::string, std::string>& profile) {
		}

		void Profiler::onProfile(const std::map<std::string, std::string>& profile) {
		}

		void Profiler::afterProfile(const std::map<std::string, std::string>& profile, const std::shared_ptr<Persistence::ProfilerSession>& profilerSession) {
		}

		Monitoring::Profiler::Profiler(
			std::string profileName,
			std::vector<std::map<std::string, std::string>> profiles,
			std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors,
			const unsigned int& runsPerProfile,
			const unsigned int& iterationsPerRun,
			const bool& randomize)
			: profileName_(std::move(profileName))
			, profiles_(std::move(profiles))
			, runsPerProfile_(runsPerProfile)
			, iterationsPerRun_(iterationsPerRun)
			, monitors_(std::move(monitors))
			, randomize_(randomize) {
		}

		std::string Profiler::getProfileName() const {
			return profileName_;
		}

		void Profiler::setProfileName(const std::string& profileName) {
			profileName_ = profileName;
		}

		std::vector<std::map<std::string, std::string>> Profiler::getProfiles() const {
			return profiles_;
		}

		void Profiler::setProfiles(const std::vector<std::map<std::string, std::string>>& profiles) {
			profiles_ = profiles;
		}

		unsigned int Profiler::getRunsPerProfile() const {
			return runsPerProfile_;
		}
		void Profiler::setRunsPerProfile(const unsigned int& runsPerProfile) {
			runsPerProfile_ = runsPerProfile;
		}

		unsigned int Profiler::getIterationsPerRun() const {
			return iterationsPerRun_;
		}

		void Profiler::setIterationsPerRun(const unsigned int& iterationsPerRun) {
			iterationsPerRun_ = iterationsPerRun;
		}

		std::vector<std::shared_ptr<Persistence::ProfilerSession>> Profiler::getProfilerSessions() const {
			return profilerSessions_;
		}

		bool Profiler::getRandomize() const {
			return randomize_;
		}

		void Profiler::setRandomize(const bool& autosave) {
			randomize_ = autosave;
		}
	}
}
#include "./Profiler.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Monitoring/Persistence/MonitorSession.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <algorithm>
#include <random>
#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Profilers {
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
					auto& profile = profiles[profileIndex];

					if(profile.find("iterations") == profile.end()) {
						profile["iterations"] = Utility::Text::toString(getIterationsPerRun());
					}

					for(unsigned int runIndex = 0; runIndex < getRunsPerProfile(); ++runIndex) {
						try {
							Utility::Logging::logInformation("Preparing profiler execution...");
							beforeProfile(profile);

							// Set up devices
							Utility::Logging::logInformation("Setting device parameters...");
							std::shared_ptr<Hardware::CPU::Core> core;
							if(profile.find("core") != profile.end()) {
								core = Hardware::CPU::Core::getCore(std::stoi(profile.at("core")));

								// Set up the defaults
								core->resetCoreClockRate();
								core->getCPU()->setTurboEnabled(true);
								core->getCPU()->resetCoreClockRate();

								// Apply custom configurations
								if(profile.find("minimumCPUClockRate") != profile.end() && profile.find("maximumCPUClockRate") != profile.end()) {
									core->getCPU()->setTurboEnabled(false);
									core->getCPU()->setCoreClockRate(std::stoul(profile.at("minimumCPUClockRate")), std::stoul(profile.at("maximumCPUClockRate")));
								}
							}

							std::shared_ptr<Hardware::GPU> gpu;
							if(profile.find("gpu") != profile.end()) {
								gpu = Hardware::GPU::getGPU(std::stoi(profile.at("gpu")));

								// Set up the defaults
								gpu->makeActive();
								gpu->reset();
								//gpu->setAutoBoostedClocksEnabled(true);
								gpu->resetCoreClockRate();

								// Apply custom configurations
								if(profile.find("gpuSynchronizationMode") != profile.end()) {
									const auto synchronizationMode = profile["gpuSynchronizationMode"];
									if(synchronizationMode == "AUTOMATIC") {
										gpu->setSynchronizationMode(Hardware::GPU::SynchronizationMode::AUTOMATIC);
									} else if(synchronizationMode == "SPIN") {
										gpu->setSynchronizationMode(Hardware::GPU::SynchronizationMode::SPIN);
									} else if(synchronizationMode == "YIELD") {
										gpu->setSynchronizationMode(Hardware::GPU::SynchronizationMode::YIELD);
									} else if(synchronizationMode == "BLOCKING") {
										gpu->setSynchronizationMode(Hardware::GPU::SynchronizationMode::BLOCKING);
									}
								}

								if(profile.find("minimumGPUClockRate") != profile.end() && profile.find("maximumGPUClockRate") != profile.end()) {
									//gpu->setAutoBoostedClocksEnabled(false);
									gpu->setCoreClockRate(std::stoul(profile.at("minimumGPUClockRate")), std::stoul(profile.at("maximumGPUClockRate")));
								}
							}

							EnergyManager::Utility::Logging::logInformation(
								"Profiling profile %d/%d, run %d/%d, with profile {%s}...",
								profileIndex + 1,
								getProfiles().size(),
								runIndex + 1,
								getRunsPerProfile(),
								Utility::Text::join(profile, ", ", " = ").c_str());
							if(!(profileIndex == 0 && runIndex == 0)) {
								const auto now = std::chrono::system_clock::now();
								const auto timeSinceStart = now - getStartTimestamp();
								const auto completedRuns = profileIndex * getRunsPerProfile() + runIndex;
								const auto timePerRun = timeSinceStart / completedRuns;
								const auto runsLeft = getProfiles().size() * getRunsPerProfile() - completedRuns;
								const auto timeToEnd = runsLeft * timePerRun;
								const auto endTime = now + timeToEnd;

								EnergyManager::Utility::Logging::logInformation("Time left: %s", Utility::Text::formatDuration(timeToEnd).c_str());
								EnergyManager::Utility::Logging::logInformation("ETA: %s", Utility::Text::formatTimestamp(endTime).c_str());
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

							// Reset the devices
							Utility::Logging::logInformation("Resetting device parameters...");
							if(core != nullptr) {
								core->getCPU()->resetCoreClockRate();
								core->getCPU()->setTurboEnabled(true);
							}

							if(gpu != nullptr) {
								gpu->resetCoreClockRate();
								//gpu->setAutoBoostedClocksEnabled(true);
								gpu->reset();
							}

							Utility::Logging::logInformation("Collecting profiler session data...");

							// Set up the session
							auto profilerSession = std::make_shared<Persistence::ProfilerSession>(getProfileName(), profile);

							// Get the monitor data and add it to the session
							std::vector<std::shared_ptr<Persistence::MonitorSession>> monitorSessions;
							for(auto& monitor : monitors_) {
								monitorSessions.push_back(std::make_shared<Persistence::MonitorSession>(monitor->getName(), monitor->getVariableValues(), profilerSession));
							}
							profilerSession->setMonitorSessions(monitorSessions);

							Utility::Logging::logInformation("Finalizing profiler execution...");
							afterProfile(profile, profilerSession);

							// Save the data
							if(getAutosave()) {
								profilerSession->save();
							}

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

			std::vector<Utility::Units::Hertz>
				Profiler::generateFrequencyValueRange(const Utility::Units::Hertz& minimumClockRate, const Utility::Units::Hertz& maximumClockRate, const unsigned int& clockRatesToProfile) {
				return generateValueRange<Utility::Units::Hertz>(minimumClockRate, maximumClockRate, clockRatesToProfile);
			}

			std::vector<Utility::Units::Hertz> Profiler::generateFrequencyValueRange(const std::shared_ptr<Hardware::Processor>& processor, const unsigned int& clockRatesToProfile) {
				return generateFrequencyValueRange(processor->getMinimumCoreClockRate().toValue(), processor->getMaximumCoreClockRate().toValue(), clockRatesToProfile);
			}

			Profiler::Profiler(
				std::string profileName,
				std::vector<std::map<std::string, std::string>> profiles,
				std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors,
				const unsigned int& runsPerProfile,
				const unsigned int& iterationsPerRun,
				const bool& randomize,
				const bool& autosave)
				: profileName_(std::move(profileName))
				, profiles_(std::move(profiles))
				, runsPerProfile_(runsPerProfile)
				, iterationsPerRun_(iterationsPerRun)
				, monitors_(std::move(monitors))
				, randomize_(randomize)
				, autosave_(autosave) {
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

			bool Profiler::getAutosave() const {
				return autosave_;
			}

			void Profiler::setAutosave(const bool& autosave) {
				autosave_ = autosave;
			}
		}
	}
}
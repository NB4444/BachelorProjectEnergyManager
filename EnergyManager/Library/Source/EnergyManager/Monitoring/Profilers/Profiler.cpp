#include "./Profiler.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Monitoring/Monitors/EARMonitor.hpp"
#include "EnergyManager/Monitoring/Persistence/MonitorSession.hpp"
#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Environment.hpp"
#include "EnergyManager/Utility/SLURM.hpp"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <unistd.h>
#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Profilers {
			void Profiler::runProfile(const std::map<std::string, std::string>& profile) {
				const unsigned int attempts = 3;

				// Retry if the profiler fails
				for(unsigned int attempt = 1; attempt <= attempts; ++attempt) {
					// This is the child
					logDebug("Preparing profiler execution...");
					beforeProfile(profile);

					// Set up devices
					logDebug("Setting device parameters...");
					std::shared_ptr<Hardware::CPU::Core> core;
					if(profile.find("core") != profile.end()) {
						core = Hardware::CPU::Core::getCore(std::stoi(profile.at("core")));

						// Set up the defaults
						if(!slurm_) {
							core->resetCoreClockRate();
							core->getCPU()->setTurboEnabled(true);
							core->getCPU()->resetCoreClockRate();
						}

						// Apply custom configurations
						if(profile.find("minimumCPUClockRate") != profile.end() && profile.find("maximumCPUClockRate") != profile.end()) {
							if(!slurm_) {
								core->getCPU()->setTurboEnabled(false);
								core->getCPU()->setCoreClockRate(std::stoul(profile.at("minimumCPUClockRate")), std::stoul(profile.at("maximumCPUClockRate")));
							}
						}
					}

					std::shared_ptr<Hardware::GPU> gpu;
					if(profile.find("gpu") != profile.end()) {
						gpu = Hardware::GPU::getGPU(std::stoi(profile.at("gpu")));

						// Set up the defaults
						if(!slurm_ && !ear_) {
							gpu->makeActive();
							gpu->reset();
							//gpu->setAutoBoostedClocksEnabled(true);
							gpu->resetCoreClockRate();
						}

						// Apply custom configurations
						if(!slurm_ && !ear_ && profile.find("gpuSynchronizationMode") != profile.end()) {
							const auto synchronizationMode = profile.at("gpuSynchronizationMode");
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
							if(!slurm_ && !ear_) {
								//gpu->setAutoBoostedClocksEnabled(false);
								gpu->setCoreClockRate(std::stoul(profile.at("minimumGPUClockRate")), std::stoul(profile.at("maximumGPUClockRate")));
							}
						}
					}

					// Start the monitor threads
					for(auto& monitor : monitors_) {
						logDebug("Starting monitor %s thread...", monitor->getName().c_str());
						monitor->reset();
						monitor->run(true);
					}

					// Profile the workload
					bool succeeded = false;
					try {
						logDebug("Profiling workload...");
						for(unsigned int iterationIndex = 0; iterationIndex < getIterationsPerRun(); ++iterationIndex) {
							onProfile(profile);
						}

						succeeded = true;
					} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
						logWarning("Failed to profile (attempt %d/%d):", attempt, attempts);
						exception.log();
					} catch(const std::exception& exception) {
						logWarning("Failed to profile (attempt %d/%d):", attempt, attempts);
						EnergyManager::Utility::Exceptions::Exception(exception.what(), __FILE__, __LINE__).log();
					} catch(...) {
						logWarning("Failed to profile (attempt %d/%d)", attempt, attempts);
					}

					// Stop all monitors threads
					for(auto& monitor : monitors_) {
						logDebug("Stopping monitor %s...", monitor->getName().c_str());
						monitor->stop();
					}

					// Wait for the monitors to finish
					for(auto& monitor : monitors_) {
						monitor->synchronize();
					}

					// Reset the devices
					logDebug("Resetting device parameters...");
					if(!slurm_ && !ear_ && core != nullptr) {
						core->getCPU()->resetCoreClockRate();
						core->getCPU()->setTurboEnabled(true);
					}

					if(!slurm_ && !ear_ && gpu != nullptr) {
						gpu->resetCoreClockRate();
						//gpu->setAutoBoostedClocksEnabled(true);
						gpu->reset();
					}

					if(succeeded) {
						logDebug("Collecting profiler session data...");

						// Set up the session
						auto profilerSession = std::make_shared<Persistence::ProfilerSession>(getProfileName(), profile);

						// Get the monitor data and add it to the session
						std::vector<std::shared_ptr<Persistence::MonitorSession>> monitorSessions;
						for(auto& monitor : monitors_) {
							monitorSessions.push_back(std::make_shared<Persistence::MonitorSession>(monitor->getName(), monitor->getVariableValues(), profilerSession));
						}
						profilerSession->setMonitorSessions(monitorSessions);

						logDebug("Finalizing profiler execution...");
						afterProfile(profile, profilerSession);

						// Save the data
						if(getAutoSave()) {
							profilerSession->save();
						}

						// Add the session to the collection
						profilerSessions_.push_back(profilerSession);

						// Break out of the retry loop
						return;
					}
				}

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Profiler failed to profile");
			}

			void Profiler::runSLURMProfile(const std::map<std::string, std::string>& profile) {
				logDebug("SLURM enabled, starting child SLURM process...");

				// Prepare application parameters
				auto applicationParameters = Utility::Text::flatten(slurmArguments_);

				// Set up the new application as a runner
				applicationParameters.push_back("--slurm-runner");

				// Serialize and add the profile
				applicationParameters.push_back("--profile");
				applicationParameters.push_back(Utility::Text::join(profile, ", ", ": "));

				// Create and run the SLURM job
				logDebug("Running SLURM job...");
				auto jobResults = Utility::SLURM::runJob(
					Utility::Environment::getApplicationPath(),
					applicationParameters,
					"EnergyManager-Profiler",
					1,
					1,
					"V100_16GB&NUMGPU2&rack26&EDR",
					true,
					profile.find("minimumCPUClockRate") == profile.end() ? Utility::Units::Hertz() : Utility::Units::Hertz(std::stod(profile.at("minimumCPUClockRate"))),
					profile.find("maximumCPUClockRate") == profile.end() ? Utility::Units::Hertz() : Utility::Units::Hertz(std::stod(profile.at("maximumCPUClockRate"))),
					profile.find("maximumGPUClockRate") == profile.end() ? Utility::Units::Hertz() : Utility::Units::Hertz(std::stod(profile.at("maximumGPUClockRate"))),
					true);

				// Start the monitor
				std::shared_ptr<Monitoring::Monitors::EARMonitor> earMonitor;
				if(ear_) {
					logDebug("Starting EAR monitor...");
					earMonitor = std::make_shared<Monitoring::Monitors::EARMonitor>(jobResults.jobID, 0, earMonitorInterval_);
					earMonitor->run(true);
				}

				// Load the profiler session (partially, for now)
				const auto profilerSession = std::make_shared<Monitoring::Persistence::ProfilerSession>(getProfileName(), std::map<std::string, std::string> {});
				std::regex regex("Profiler session ID: (\\d+)");
				std::smatch match;
				if(std::regex_search(jobResults.output, match, regex)) {
					unsigned int profilerSessionID = std::stoul(match.str(1));

					logDebug("Found profiler session ID: %d", profilerSessionID);

					// Add the session
					profilerSession->setID(profilerSessionID);
				} else {
					logWarning("Could not retrieve session ID from output, EAR data will not be included");
				}
				profilerSessions_.push_back(profilerSession);

				// Stop the EAR monitor
				if(ear_) {
					logDebug("Stopping EAR monitor...");
					earMonitor->stop(true);

					// Append the monitor data to the session
					auto& profilerSession = profilerSessions_.back();

					if(profilerSession->getID() >= 0) {
						logDebug("Storing EAR monitor data in profiler session %d...", profilerSession->getID());

						// Get the monitor data and add it to the session
						profilerSession->setMonitorSessions({ std::make_shared<Persistence::MonitorSession>(earMonitor->getName(), earMonitor->getVariableValues(), profilerSession) });

						// Save the updated session
						profilerSession->save();
					}
				}
			}

			std::vector<std::string> Profiler::generateHeaders() const {
				auto headers = Runnable::generateHeaders();
				headers.push_back("Profiler " + getProfileName());

				return headers;
			}

			void Profiler::beforeRun() {
				profilerSessions_.clear();
			}

			void Profiler::onRun() {
				// Show information about the current node
				if(slurm_) {
					Utility::SLURM::logInformation();
				}

				// Check if this is the parent process
				if(slurmArguments_.find("--slurm-runner") == slurmArguments_.end()) {
					// This is the parent
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

						logInformation("Profiling profile %d/%d: {%s}...", profileIndex + 1, getProfiles().size(), Utility::Text::join(profile, ", ", " = ").c_str());

						if(profile.find("iterations") == profile.end()) {
							profile["iterations"] = Utility::Text::toString(getIterationsPerRun());
						}

						for(unsigned int runIndex = 0; runIndex < getRunsPerProfile(); ++runIndex) {
							logInformation("Profiling run %d/%d...", runIndex + 1, getRunsPerProfile());
							if(!(profileIndex == 0 && runIndex == 0)) {
								const auto now = std::chrono::system_clock::now();
								const auto timeSinceStart = now - getStartTimestamp();
								const auto completedRuns = profileIndex * getRunsPerProfile() + runIndex;
								const auto timePerRun = timeSinceStart / completedRuns;
								const auto runsLeft = getProfiles().size() * getRunsPerProfile() - completedRuns;
								const auto timeToEnd = runsLeft * timePerRun;
								const auto endTime = now + timeToEnd;

								logInformation("Time left: %s", Utility::Text::formatDuration(timeToEnd).c_str());
								logInformation("ETA: %s", Utility::Text::formatTimestamp(endTime).c_str());
							}

							if(slurm_) {
								runSLURMProfile(profile);
							} else {
								// Simply run the profile if slurm is disabled
								runProfile(profile);
							}
						}
					}
				} else {
					logDebug("Initializing SLURM runner...");

					// Deserialize the profile
					std::map<std::string, std::string> profile = Utility::Text::splitToMap(Utility::Text::getArgument<std::string>(slurmArguments_, "--profile", ""), ", ", ": ");
					logDebug("Profile: {%s}", Utility::Text::join(profile, ", ", ": ").c_str());

					// Run the profile
					logDebug("Running SLURM profile...");
					runProfile(profile);

					// Signal the profiler session ID to the parent process
					logInformation("Profiler session ID: %d", profilerSessions_.back()->getID());
				}

				logInformation("Finished profiling profiles");
			}

			void Profiler::beforeProfile(const std::map<std::string, std::string>& profile) {
			}

			void Profiler::onProfile(const std::map<std::string, std::string>& profile) {
			}

			void Profiler::afterProfile(const std::map<std::string, std::string>& profile, const std::shared_ptr<Persistence::ProfilerSession>& profilerSession) {
			}

			Profiler::Profiler(
				std::string profileName,
				std::vector<std::map<std::string, std::string>> profiles,
				std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors,
				const unsigned int& runsPerProfile,
				const unsigned int& iterationsPerRun,
				const bool& randomize,
				const bool& autoSave,
				const bool& slurm,
				const std::map<std::string, std::string>& slurmArguments,
				const bool& ear,
				const std::chrono::system_clock::duration& earMonitorInterval)
				: profileName_(std::move(profileName))
				, profiles_(std::move(profiles))
				, runsPerProfile_(runsPerProfile)
				, iterationsPerRun_(iterationsPerRun)
				, monitors_(std::move(monitors))
				, randomize_(randomize)
				, autoSave_(autoSave)
				, slurm_(slurm)
				, slurmArguments_(slurmArguments)
				, ear_(ear)
				, earMonitorInterval_(earMonitorInterval) {
			}

			Profiler::Profiler(const std::string& profileName, const std::vector<std::map<std::string, std::string>>& profiles, const std::map<std::string, std::string>& arguments)
				: runsPerProfile_(Utility::Text::getArgument<unsigned int>(arguments, "--runsPerProfile", 1))
				, iterationsPerRun_(Utility::Text::getArgument<unsigned int>(arguments, "--iterationsPerRun", 3))
				, randomize_(Utility::Text::getArgument<bool>(arguments, "--randomize", true))
				, autoSave_(Utility::Text::getArgument<bool>(arguments, "--autoSave", true))
				, slurm_(Utility::Text::getArgument<bool>(arguments, "--slurm", true))
				, slurmArguments_(arguments)
				, ear_(Utility::Text::getArgument<bool>(arguments, "--ear", true)) {
				// Get hardware
				static const auto core = Hardware::CPU::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
				static const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

				// Configure monitors
				static const auto monitorInterval = Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(100));

				// Determine if this is control data
				static const auto control = Utility::Text::getArgument<bool>(arguments, "--control", true);

				// Configure the Profiler
				profileName_ = std::string(control ? "" : "Fixed Frequency ") + profileName;
				profiles_ = control ? profiles
									: Profiler::generateFixedFrequencyProfiles(
										profiles,
										core,
										Utility::Text::getArgument<unsigned int>(arguments, "--cpuCoreClockRatesToProfile", 5),
										gpu,
										Utility::Text::getArgument<unsigned int>(arguments, "--gpuCoreClockRatesToProfile", 5));
				monitors_ = Monitors::Monitor::getMonitorsForAllDevices(
					Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--applicationMonitorInterval", monitorInterval),
					Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--nodeMonitorInterval", monitorInterval),
					Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--cpuMonitorInterval", monitorInterval),
					Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--cpuCoreMonitorInterval", monitorInterval),
					Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--gpuMonitorInterval", monitorInterval));
				earMonitorInterval_ = Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--earMonitorInterval", monitorInterval);
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

			bool Profiler::getAutoSave() const {
				return autoSave_;
			}

			void Profiler::setAutoSave(const bool& autoSave) {
				autoSave_ = autoSave;
			}
		}
	}
}
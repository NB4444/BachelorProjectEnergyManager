#include "./Profiler.hpp"

#include "EnergyManager/Configuration.hpp"
#include "EnergyManager/Monitoring/Monitors/EARMonitor.hpp"
#include "EnergyManager/Monitoring/Persistence/MonitorSession.hpp"
#include "EnergyManager/Utility/Environment.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/SLURM.hpp"

#include <algorithm>
#include <random>
#include <utility>

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			void Profiler::runProfile(const std::map<std::string, std::string>& profile, const unsigned int& attempts) {
				// This is the child
				logDebug("Preparing profiler execution...");
				beforeProfile(profile);

				// Set up devices
				logDebug("Setting device parameters...");
				std::shared_ptr<Hardware::Core> core;
				if(profile.find("core") != profile.end()) {
					core = Hardware::Core::getCore(std::stoi(profile.at("core")));

					// Set up the defaults
#ifndef SLURM_ENABLED
					core->resetCoreClockRate();
					core->getCPU()->setTurboEnabled(true);
					core->getCPU()->resetCoreClockRate();
#endif

					// Apply custom configurations
					if(profile.find("minimumCPUClockRate") != profile.end() && profile.find("maximumCPUClockRate") != profile.end()) {
#ifndef SLURM_ENABLED
						core->getCPU()->setTurboEnabled(false);
						core->getCPU()->setCoreClockRate(std::stoul(profile.at("minimumCPUClockRate")), std::stoul(profile.at("maximumCPUClockRate")));
#endif
					}
				}

				std::shared_ptr<Hardware::GPU> gpu;
				if(profile.find("gpu") != profile.end()) {
					gpu = Hardware::GPU::getGPU(std::stoi(profile.at("gpu")));

					// Set up the defaults
					gpu->makeActive();
#if !defined(SLURM_ENABLED) && !defined(EAR_ENABLED)
					gpu->reset();
					//gpu->setAutoBoostedClocksEnabled(true);
					gpu->resetCoreClockRate();
#endif

					// Apply custom configurations
#if !defined(SLURM_ENABLED) && !defined(EAR_ENABLED)
					if(profile.find("gpuSynchronizationMode") != profile.end()) {
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
#endif

					if(profile.find("minimumGPUClockRate") != profile.end() && profile.find("maximumGPUClockRate") != profile.end()) {
#if !defined(SLURM_ENABLED) && !defined(EAR_ENABLED)
						//gpu->setAutoBoostedClocksEnabled(false);
						gpu->setCoreClockRate(std::stoul(profile.at("minimumGPUClockRate")), std::stoul(profile.at("maximumGPUClockRate")));
#endif
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
						logInformation("Profiling iteration %d/%d...", iterationIndex + 1, getIterationsPerRun());
						onProfile(profile);
					}

					succeeded = true;
				} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
					exception.log();
					logWarning("Failed to profile");
				} catch(const std::exception& exception) {
					EnergyManager::Utility::Exceptions::Exception(exception.what(), __FILE__, __LINE__).log();
					logWarning("Failed to profile");
				} catch(...) {
					logWarning("Failed to profile");
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
#if !defined(SLURM_ENABLED) && !defined(EAR_ENABLED)
				if(core != nullptr) {
					core->getCPU()->resetCoreClockRate();
					core->getCPU()->setTurboEnabled(true);
				}

				if(gpu != nullptr) {
					gpu->resetCoreClockRate();
					//gpu->setAutoBoostedClocksEnabled(true);
					gpu->reset();
				}
#endif

				if(succeeded) {
					logDebug("Setting up profiler session...");

					// Set up the session
					auto profilerSession = std::make_shared<Persistence::ProfilerSession>(getProfileName(), profile);

					logTrace("Collecting profiler session data...");

					// Get the monitor data and add it to the session
					std::vector<std::shared_ptr<Monitoring::Persistence::MonitorSession>> monitorSessions;
					for(auto& monitor : monitors_) {
						logTrace("Adding monitor data...");

						monitorSessions.push_back(std::make_shared<Monitoring::Persistence::MonitorSession>(monitor->getName(), monitor->getVariableValues(), profilerSession));
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

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Profiler failed to profile");
			}

			void Profiler::runSLURMProfile(const std::map<std::string, std::string>& profile, const unsigned int& attempts) {
				logDebug("SLURM enabled, starting child SLURM process...");

				// Prepare application parameters
				auto applicationParameters = Utility::Text::flatten(slurmArguments_);

				// Set up the new application as a runner
				applicationParameters.push_back("--slurmRunner");

				// Serialize and add the profile
				applicationParameters.push_back("--profileName");
				applicationParameters.push_back(getProfileName());
				applicationParameters.push_back("--profile");
				applicationParameters.push_back(Utility::Text::join(profile, ", ", ": "));

				// Create and run the SLURM job
				auto applicationPath = Utility::Environment::getApplicationPath();
				bool succeeded = false;
				unsigned int jobID;
				std::string output;
				try {
					logDebug("Running SLURM job...");
					auto jobResults = Utility::SLURM::runJob(
						applicationPath,
						applicationParameters,
						"EnergyManager-Profiler",
						1,
						1,
						"V100_16GB&NUMGPU2&rack26&EDR",
						true,
						profile.find("minimumCPUClockRate") == profile.end() ? Utility::Units::Hertz() : Utility::Units::Hertz(std::stod(profile.at("minimumCPUClockRate"))),
						profile.find("maximumCPUClockRate") == profile.end() ? Utility::Units::Hertz() : Utility::Units::Hertz(std::stod(profile.at("maximumCPUClockRate"))),
						-1,
						profile.find("maximumGPUClockRate") == profile.end() ? Utility::Units::Hertz() : Utility::Units::Hertz(std::stod(profile.at("maximumGPUClockRate"))),
						std::chrono::milliseconds(50),
						true);

					logDebug("Retrieving SLURM job results...");
					jobID = jobResults.jobID;
					output = jobResults.output;

					succeeded = true;
				} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
					exception.log();
					logWarning("Failed to profile");
				} catch(const std::exception& exception) {
					EnergyManager::Utility::Exceptions::Exception(exception.what(), __FILE__, __LINE__).log();
					logWarning("Failed to profile");
				} catch(...) {
					logWarning("Failed to profile");
				}

				if(succeeded) {
#ifdef EAR_ENABLED
					logDebug("Starting EAR monitor...");
					// Start the monitor
					std::shared_ptr<Monitoring::Monitors::EARMonitor> earMonitor = std::make_shared<Monitoring::Monitors::EARMonitor>(jobID, 0, earMonitorInterval_);
					earMonitor->run(true);
#endif

					logTrace("Job output: %s", output.c_str());

					// Load the profiler session (partially, for now)
					const auto profilerSession = std::make_shared<Profiling::Persistence::ProfilerSession>(getProfileName(), std::map<std::string, std::string> {});
					std::regex regex("Profiler session ID: (\\d+)");
					std::smatch match;
					if(std::regex_search(output, match, regex)) {
						unsigned int profilerSessionID = std::stoul(match.str(1));

						logDebug("Found profiler session ID: %d", profilerSessionID);

						// Add the session
						profilerSession->setID(profilerSessionID);
					} else {
						logWarning("Could not retrieve session ID from output, EAR data will not be included");
					}
					profilerSessions_.push_back(profilerSession);

					// Stop the EAR monitor
#ifdef EAR_ENABLED
					logDebug("Stopping EAR monitor...");
					earMonitor->stop(true);

					logDebug("Accessing profiler session...");
					if(profilerSession->getID() >= 0) {
						logDebug("Storing EAR monitor data in profiler session %d...", profilerSession->getID());

						// Get the monitor data and add it to the session
						profilerSession->setMonitorSessions({ std::make_shared<Monitoring::Persistence::MonitorSession>(earMonitor->getName(), earMonitor->getVariableValues(), profilerSession) });

						// Save the updated session
						profilerSession->save();
					}
#endif

					// Break out of the retry loop
					return;
				}

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Profiler failed to profile through SLURM");
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
#ifdef SLURM_ENABLED
				logInformation("Program started with the following arguments:\n%s", Utility::Text::join(slurmArguments_, "\n", "=").c_str());
				// FIXME: This causes a crash for some reason
				//Utility::SLURM::logInformation();
#endif

				// Check if this is the parent process
				if(slurmArguments_.find("--slurmRunner") == slurmArguments_.end()) { // This is the parent
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

							// Retry if the profiler fails
							Utility::Exceptions::Exception::retry(
								[&] {
#ifdef SLURM_ENABLED
									runSLURMProfile(profile);
#else
									// Simply run the profile if slurm is disabled
									runProfile(profile);
#endif
								},
								retriesPerRun_);
						}
					}
				} else {
					logDebug("Initializing SLURM runner...");

					// Get the profile name
					std::string profileName = Utility::Text::getArgument<std::string>(slurmArguments_, "--profileName", "");
					logDebug("Profile name: %s", profileName.c_str());

					// Check if this is the correct profiler
					if(getProfileName() != profileName) {
						// If not, stop processing
						logDebug("Skipping profiler %s...", getProfileName().c_str());
						return;
					}

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

			std::vector<Utility::Units::Hertz>
				Profiler::generateFrequencyValueRange(const Utility::Units::Hertz& minimumClockRate, const Utility::Units::Hertz& maximumClockRate, const unsigned int& clockRatesToProfile) {
				return generateValueRange<Utility::Units::Hertz>(minimumClockRate, maximumClockRate, clockRatesToProfile);
			}

			std::vector<Utility::Units::Hertz> Profiler::generateFrequencyValueRange(const std::shared_ptr<Hardware::Processor>& processor, const unsigned int& clockRatesToProfile) {
				return generateFrequencyValueRange(processor->getMinimumCoreClockRate().toValue(), processor->getMaximumCoreClockRate().toValue(), clockRatesToProfile);
			}

			std::vector<std::map<std::string, std::string>> Profiler::generateFixedFrequencyProfiles(
				const std::vector<std::map<std::string, std::string>>& profiles,
				const std::shared_ptr<Hardware::Core>& core,
				const unsigned int& coreClockRatesToProfile,
				const std::shared_ptr<Hardware::GPU>& gpu,
				const unsigned int& gpuClockRatesToProfile) {
				std::vector<std::map<std::string, std::string>> results;
				for(const auto& coreClockRate : Profiler::generateFrequencyValueRange(core, coreClockRatesToProfile)) {
					for(const auto& gpuClockRate : Profiler::generateFrequencyValueRange(gpu->getMinimumCoreClockRate(), gpu->getMaximumCoreClockRate(), gpuClockRatesToProfile)) {
						for(const auto& profile : profiles) {
							// Generate a new profile with the frequencies set
							std::map<std::string, std::string> newProfile = {
								{ "minimumCPUClockRate", Utility::Text::toString(coreClockRate) },
								{ "maximumCPUClockRate", Utility::Text::toString(coreClockRate) },
								{ "minimumGPUClockRate", Utility::Text::toString(gpuClockRate) },
								{ "maximumGPUClockRate", Utility::Text::toString(gpuClockRate) },
							};

							// Append the current profile
							newProfile.insert(profile.begin(), profile.end());
							results.push_back(newProfile);
						}
					}
				}

				return results;
			}

			Profiler::Profiler(
				std::string profileName,
				std::vector<std::map<std::string, std::string>> profiles,
				std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors,
				const unsigned int& runsPerProfile,
				const unsigned int& iterationsPerRun,
				const unsigned int& retriesPerRun,
				const bool& randomize,
				const bool& autoSave,
				const std::map<std::string, std::string>& slurmArguments,
				const std::chrono::system_clock::duration& earMonitorInterval)
				: profileName_(std::move(profileName))
				, profiles_(std::move(profiles))
				, runsPerProfile_(runsPerProfile)
				, iterationsPerRun_(iterationsPerRun)
				, retriesPerRun_(retriesPerRun)
				, monitors_(std::move(monitors))
				, randomize_(randomize)
				, autoSave_(autoSave)
				, slurmArguments_(slurmArguments)
				, earMonitorInterval_(earMonitorInterval) {
			}

			Profiler::Profiler(const std::string& profileName, const std::vector<std::map<std::string, std::string>>& profiles, const std::map<std::string, std::string>& arguments)
				: runsPerProfile_(Utility::Text::getArgument<unsigned int>(arguments, "--runsPerProfile", 1))
				, iterationsPerRun_(Utility::Text::getArgument<unsigned int>(arguments, "--iterationsPerRun", 1))
				, retriesPerRun_(Utility::Text::getArgument<unsigned int>(arguments, "--retriesPerRun", 10))
				, randomize_(Utility::Text::getArgument<bool>(arguments, "--randomize", true))
				, autoSave_(Utility::Text::getArgument<bool>(arguments, "--autoSave", true))
				, slurmArguments_(arguments) {
				// Get hardware
				const auto core = Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
				const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

				// Configure monitors
				const auto monitorInterval = Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(250));

				// Determine if the profiles should be profiled using a set of CPU and GPU frequencies
				const auto fixedClockRates = Utility::Text::getArgument<bool>(arguments, "--fixedClockRates", false);

				// Configure the Profiler
				profileName_ = std::string(fixedClockRates ? "Fixed Frequency " : "") + profileName;
				profiles_ = fixedClockRates ? Profiler::generateFixedFrequencyProfiles(
								profiles,
								core,
								Utility::Text::getArgument<unsigned int>(arguments, "--cpuCoreClockRatesToProfile", 5),
								gpu,
								Utility::Text::getArgument<unsigned int>(arguments, "--gpuCoreClockRatesToProfile", 5))
											: profiles;
				monitors_ = Monitoring::Monitors::Monitor::getMonitorsForAllDevices(
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

			void Profiler::addMonitor(const std::shared_ptr<Monitoring::Monitors::Monitor>& monitor) {
				monitors_.push_back(monitor);
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
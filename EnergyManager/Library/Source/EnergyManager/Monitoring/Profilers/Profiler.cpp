#include "./Profiler.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Monitoring/Persistence/MonitorSession.hpp"
#include "EnergyManager/Testing/Application.hpp"
#include "EnergyManager/Utility/Environment.hpp"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <utility>

namespace EnergyManager {
	namespace Monitoring {
		namespace Profilers {
			void Profiler::runProfile(const std::map<std::string, std::string>& profile) {
				// This is the child
				logDebug("Preparing profiler execution...");
				beforeProfile(profile);

				// Set up devices
				logDebug("Setting device parameters...");
				std::shared_ptr<Hardware::CPU::Core> core;
				if(profile.find("core") != profile.end()) {
					core = Hardware::CPU::Core::getCore(std::stoi(profile.at("core")));

					// Set up the defaults
					if(!ear_) {
						core->resetCoreClockRate();
						core->getCPU()->setTurboEnabled(true);
						core->getCPU()->resetCoreClockRate();
					}

					// Apply custom configurations
					if(profile.find("minimumCPUClockRate") != profile.end() && profile.find("maximumCPUClockRate") != profile.end()) {
						if(!ear_) {
							core->getCPU()->setTurboEnabled(false);
							core->getCPU()->setCoreClockRate(std::stoul(profile.at("minimumCPUClockRate")), std::stoul(profile.at("maximumCPUClockRate")));
						}
					}
				}

				std::shared_ptr<Hardware::GPU> gpu;
				if(profile.find("gpu") != profile.end()) {
					gpu = Hardware::GPU::getGPU(std::stoi(profile.at("gpu")));

					// Set up the defaults
					if(!ear_) {
						gpu->makeActive();
						gpu->reset();
						//gpu->setAutoBoostedClocksEnabled(true);
						gpu->resetCoreClockRate();
					}

					// Apply custom configurations
					if(!ear_ && profile.find("gpuSynchronizationMode") != profile.end()) {
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
						if(!ear_) {
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
				logDebug("Profiling workload...");
				for(unsigned int iterationIndex = 0; iterationIndex < getIterationsPerRun(); ++iterationIndex) {
					onProfile(profile);
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
				if(core != nullptr) {
					core->getCPU()->resetCoreClockRate();
					core->getCPU()->setTurboEnabled(true);
				}

				if(gpu != nullptr) {
					gpu->resetCoreClockRate();
					//gpu->setAutoBoostedClocksEnabled(true);
					gpu->reset();
				}

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
				if(getAutosave()) {
					profilerSession->save();
				}

				// Add the session to the collection
				profilerSessions_.push_back(profilerSession);
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
								logDebug("SLURM enabled, starting child SLURM process...");

								// Get the path to SLURM
								const std::string slurmPath = "srun";

								// Prepare the SLURM parameters
								std::vector<std::string> slurmParameters;

								// Set the name
								slurmParameters.push_back("--job-name");
								slurmParameters.push_back("EnergyManager");

								// Set the time limit
								slurmParameters.push_back("--time");
								slurmParameters.push_back("1:00:00");

								// Ensure exclusive access
								slurmParameters.push_back("--exclusive");

								// Set the partition
								slurmParameters.push_back("--partition");
								slurmParameters.push_back("standard");

								// Set the account
								slurmParameters.push_back("--account");
								slurmParameters.push_back("COLBSC");

								// Configure amount of nodes
								slurmParameters.push_back("--nodes");
								slurmParameters.push_back("1");

								// Set the amount of repetitions per node
								slurmParameters.push_back("--ntasks-per-node");
								slurmParameters.push_back("1");

								// Constrain the nodes from which to select by using features
								slurmParameters.push_back("--constraint");
								slurmParameters.push_back("2666MHz&GPU&V100_16GB");

								// Set up the output file
								slurmParameters.push_back("-o");
								slurmParameters.push_back("log.%j.out");

								// Set up the error output file
								slurmParameters.push_back("-e");
								slurmParameters.push_back("log.%j.err");

								// Get the path to the application
								// TODO
								const std::string applicationPath;

								// Prepare application parameters
								auto applicationParameters = Utility::Text::flatten(slurmArguments_);

								// Set up the new application as a runner
								applicationParameters.push_back("--slurm-runner");

								// Serialize and add the profile
								applicationParameters.push_back("--profile");
								applicationParameters.push_back(Utility::Text::join(profile, ", ", ": "));

								// Configure the EAR framework
								if(ear_) {
									// Inject EAR into SLURM
									Utility::Environment::setVariable("SLURM_HACK_LIBRARY_FILE", "${EAR_INSTALL_PATH}/lib/libear.seq.so");

									// Disable loading MPI
									Utility::Environment::setVariable("SLURM_LOADER_LOAD_NO_MPI_LIB", applicationPath);

									// Set the verbosity
									slurmParameters.push_back("--ear-verbose");
									slurmParameters.push_back("1");

									// Set the policy
									slurmParameters.push_back("--ear-policy");
									slurmParameters.push_back("monitoring");

									// Set the CPU frequency
									if(profile.find("maximumCPUClockRate") != profile.end()) {
										slurmParameters.push_back("--cpu-freq");
										slurmParameters.push_back(
											Utility::Text::toString(Utility::Units::Hertz(std::stoul(profile.at("maximumCPUClockRate"))).convertPrefix(Utility::Units::SIPrefix::MEGA)));
									}

									// Set the GPU frequency
									if(profile.find("maximumGPUClockRate") != profile.end()) {
										Utility::Environment::setVariable(
											"SLURM_EAR_GPU_DEF_FREQ",
											Utility::Text::toString(Utility::Units::Hertz(std::stoul(profile.at("maximumCPUClockRate"))).convertPrefix(Utility::Units::SIPrefix::MEGA)));
									}
								}

								// Combine the parameters
								std::vector<std::string> parameters(slurmParameters.begin(), slurmParameters.end());
								parameters.push_back(applicationPath);
								parameters.insert(parameters.end(), applicationParameters.begin(), applicationParameters.end());

								// Prepare and run the SLURM runner
								Testing::Application(slurmPath, parameters).run(false);
							} else {
								// SLURM disabled
								// Simply run the profile if slurm is disabled
								runProfile(profile);
							}
						}
					}
				} else {
					logDebug("Initializing SLURM process...");

					// TODO: Deserialize profile and run it
					//doRun(profile);
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
				const bool& autosave,
				const bool& slurm,
				const std::map<std::string, std::string>& slurmArguments,
				const bool& ear)
				: profileName_(std::move(profileName))
				, profiles_(std::move(profiles))
				, runsPerProfile_(runsPerProfile)
				, iterationsPerRun_(iterationsPerRun)
				, monitors_(std::move(monitors))
				, randomize_(randomize)
				, autosave_(autosave)
				, slurm_(slurm)
				, slurmArguments_(slurmArguments)
				, ear_(ear) {
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
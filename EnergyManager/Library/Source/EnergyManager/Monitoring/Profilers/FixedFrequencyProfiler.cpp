#include "./FixedFrequencyProfiler.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Profilers {
			void FixedFrequencyProfiler::beforeProfile(const std::map<std::string, std::string>& profile) {
				const auto core = Hardware::CPU::Core::getCore(std::stoi(profile.at("core")));
				core->getCPU()->setCoreClockRate(std::stoul(profile.at("minimumCPUClockRate")), std::stoul(profile.at("maximumCPUClockRate")));
				core->getCPU()->setTurboEnabled(false);

				const auto gpu = Hardware::GPU::getGPU(std::stoi(profile.at("gpu")));
				gpu->setCoreClockRate(std::stoul(profile.at("minimumGPUClockRate")), std::stoul(profile.at("maximumGPUClockRate")));
				//gpu->setAutoBoostedClocksEnabled(false);
			}

			void FixedFrequencyProfiler::afterProfile(const std::map<std::string, std::string>& profile, const std::shared_ptr<Persistence::ProfilerSession>& profilerSession) {
				const auto core = Hardware::CPU::Core::getCore(std::stoi(profile.at("core")));
				core->getCPU()->resetCoreClockRate();
				core->getCPU()->setTurboEnabled(true);

				const auto gpu = Hardware::GPU::getGPU(std::stoi(profile.at("gpu")));
				gpu->resetCoreClockRate();
				//gpu->setAutoBoostedClocksEnabled(true);
			}

			FixedFrequencyProfiler::FixedFrequencyProfiler(
				const std::string& profileName,
				const std::shared_ptr<Hardware::CPU::Core>& core,
				const unsigned int& coreClockRatesToProfile,
				const std::shared_ptr<Hardware::GPU>& gpu,
				const unsigned int& gpuClockRatesToProfile,
				const std::vector<std::map<std::string, std::string>>& profiles,
				const std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>>& monitors,
				const unsigned int& runsPerProfile,
				const unsigned int& iterationsPerRun,
				const bool& randomize,
				const bool& autosave)
				: Profiler(
					profileName,
					[&] {
						std::vector<std::map<std::string, std::string>> results;
						for(const auto& coreClockRate : Profiler::generateFrequencyValueRange(core, coreClockRatesToProfile)) {
							for(const auto& gpuClockRate : Profiler::generateFrequencyValueRange(1000, gpu->getMaximumCoreClockRate(), gpuClockRatesToProfile)) {
								for(const auto& profile : profiles) {
									// Generate a new profile with the frequencies set
									std::map<std::string, std::string> newProfile = {
										{ "core", Utility::Text::toString(core->getID()) },
										{ "gpu", Utility::Text::toString(gpu->getID()) },
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
					}(),
					monitors,
					runsPerProfile,
					iterationsPerRun,
					randomize,
					autosave) {
			}
		}
	}
}
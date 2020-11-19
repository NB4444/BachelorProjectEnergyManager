#include "./FixedFrequencyProfiler.hpp"

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Profilers {
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
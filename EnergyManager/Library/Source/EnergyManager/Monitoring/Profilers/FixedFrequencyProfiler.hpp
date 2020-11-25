#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Monitoring/Persistence/ProfilerSession.hpp"
#include "EnergyManager/Monitoring/Profilers/Profiler.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Profilers {
			/**
			 * Profiles the workload at a set of fixed frequencies.
			 */
			class FixedFrequencyProfiler : public Profiler {
				using Profiler::Profiler;

			public:
				/**
				 * Creates a new FixedFrequencyProfiler.
				 * @param profileName The name of the profile.
				 * @param core The processor to use.
				 * @param clockRatesToProfile The amount of clock rates to test in between the minimum and maximum clock rate (inclusive).
				 * @param gpu The GPU to use.
				 * @param gpuClockRatesToProfile The amount of clock rates to test in between the minimum and maximum clock rate (inclusive).
				 * @param slurm Whether to use SLURM.
				 * @param slurmArguments The arguments to use for SLURM.
				 */
				FixedFrequencyProfiler(
					const std::string& profileName,
					const std::shared_ptr<Hardware::CPU::Core>& core,
					const unsigned int& coreClockRatesToProfile,
					const std::shared_ptr<Hardware::GPU>& gpu,
					const unsigned int& gpuClockRatesToProfile,
					const std::vector<std::map<std::string, std::string>>& profiles,
					const std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>>& monitors,
					const unsigned int& runsPerProfile = 1,
					const unsigned int& iterationsPerRun = 1,
					const bool& randomize = false,
					const bool& autosave = false,
					const bool& slurm = false,
					const std::map<std::string, std::string>& slurmArguments = {});
			};
		}
	}
}
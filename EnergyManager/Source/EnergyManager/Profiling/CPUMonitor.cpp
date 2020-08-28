#include "./CPUMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

#define ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD(KEY, VALUE) ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(cpuResults[KEY] = VALUE);

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> CPUMonitor::onPoll() {
			if(!startTimespansMeasured_) {
				startTimespansMeasured_ = true;
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startUserTimespan_ = cpu_->getUserTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startNiceTimespan_ = cpu_->getNiceTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startSystemTimespan_ = cpu_->getSystemTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startIdleTimespan_ = cpu_->getIdleTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startIOWaitTimespan_ = cpu_->getIOWaitTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startInterruptsTimespan_ = cpu_->getInterruptsTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startSoftInterruptsTimespan_ = cpu_->getSoftInterruptsTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startStealTimespan_ = cpu_->getStealTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startGuestTimespan_ = cpu_->getGuestTimespan());
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startGuestNiceTimespan_ = cpu_->getGuestNiceTimespan());

				for(unsigned int core = 0; core < cpu_->getCoreCount(); ++core) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreUserTimespans_[core] = cpu_->getUserTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreNiceTimespans_[core] = cpu_->getNiceTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreSystemTimespans_[core] = cpu_->getSystemTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreIdleTimespans_[core] = cpu_->getIdleTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreIOWaitTimespans_[core] = cpu_->getIOWaitTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreInterruptsTimespans_[core] = cpu_->getInterruptsTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreSoftInterruptsTimespans_[core] = cpu_->getSoftInterruptsTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreStealTimespans_[core] = cpu_->getStealTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreGuestTimespans_[core] = cpu_->getGuestTimespan(core));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startCoreGuestNiceTimespans_[core] = cpu_->getGuestNiceTimespan(core));
				}
			}

			std::map<std::string, std::string> cpuResults;
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("userTimespan", std::to_string((cpu_->getUserTimespan() - startUserTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("niceTimespan", std::to_string((cpu_->getNiceTimespan() - startNiceTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("systemTimespan", std::to_string((cpu_->getSystemTimespan() - startSystemTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("idleTimespan", std::to_string((cpu_->getIdleTimespan() - startIdleTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("ioWaitTimespan", std::to_string((cpu_->getIOWaitTimespan() - startIOWaitTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("interruptsTimespan", std::to_string((cpu_->getInterruptsTimespan() - startInterruptsTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("softInterruptsTimespan", std::to_string((cpu_->getSoftInterruptsTimespan() - startSoftInterruptsTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("stealTimespan", std::to_string((cpu_->getStealTimespan() - startStealTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("guestTimespan", std::to_string((cpu_->getGuestTimespan() - startGuestTimespan_).count()));
			ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("guestNiceTimespan", std::to_string((cpu_->getGuestNiceTimespan() - startGuestNiceTimespan_).count()));

			// Add the per-core values
			for(unsigned int core = 0; core < cpu_->getCoreCount(); ++core) {
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("coreClockRateCore" + std::to_string(core), std::to_string(cpu_->getCoreClockRate(core).toValue()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("coreUtilizationRateCore" + std::to_string(core), std::to_string(cpu_->getCoreUtilizationRate(core).toCombined()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("maximumCoreClockRateCore" + std::to_string(core), std::to_string(cpu_->getMaximumCoreClockRate(core).toValue()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("userTimespanCore" + std::to_string(core), std::to_string((cpu_->getUserTimespan(core) - startCoreUserTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("niceTimespanCore" + std::to_string(core), std::to_string((cpu_->getNiceTimespan(core) - startCoreNiceTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("systemTimespanCore" + std::to_string(core), std::to_string((cpu_->getSystemTimespan(core) - startCoreSystemTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("idleTimespanCore" + std::to_string(core), std::to_string((cpu_->getIdleTimespan(core) - startCoreIdleTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("ioWaitTimespanCore" + std::to_string(core), std::to_string((cpu_->getIOWaitTimespan(core) - startCoreIOWaitTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD(
					"interruptsTimespanCore" + std::to_string(core),
					std::to_string((cpu_->getInterruptsTimespan(core) - startCoreInterruptsTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD(
					"softInterruptsTimespanCore" + std::to_string(core),
					std::to_string((cpu_->getSoftInterruptsTimespan(core) - startCoreSoftInterruptsTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("stealTimespanCore" + std::to_string(core), std::to_string((cpu_->getStealTimespan(core) - startCoreStealTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD("guestTimespanCore" + std::to_string(core), std::to_string((cpu_->getGuestTimespan(core) - startCoreGuestTimespans_[core]).count()));
				ENERGY_MANAGER_PROFILING_CPU_MONITOR_ADD(
					"guestNiceTimespanCore" + std::to_string(core),
					std::to_string((cpu_->getGuestNiceTimespan(core) - startCoreGuestNiceTimespans_[core]).count()));
			}

			// Get upstream values
			auto processorResults = ProcessorMonitor::onPoll();
			cpuResults.insert(processorResults.begin(), processorResults.end());

			return cpuResults;
		}

		CPUMonitor::CPUMonitor(const std::shared_ptr<Hardware::CPU>& cpu) : ProcessorMonitor("CPUMonitor", cpu), cpu_(cpu) {
		}
	}
}
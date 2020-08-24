#include "./CPUMonitor.hpp"

namespace EnergyManager {
	namespace Profiling {
		std::map<std::string, std::string> CPUMonitor::onPoll() {
			if(!startTimespansMeasured_) {
				startTimespansMeasured_ = true;
				startUserTimespan_ = cpu_->getUserTimespan();
				startNiceTimespan_ = cpu_->getNiceTimespan();
				startSystemTimespan_ = cpu_->getSystemTimespan();
				startIdleTimespan_ = cpu_->getIdleTimespan();
				startIOWaitTimespan_ = cpu_->getIOWaitTimespan();
				startInterruptsTimespan_ = cpu_->getInterruptsTimespan();
				startSoftInterruptsTimespan_ = cpu_->getSoftInterruptsTimespan();
				startStealTimespan_ = cpu_->getStealTimespan();
				startGuestTimespan_ = cpu_->getGuestTimespan();
				startGuestNiceTimespan_ = cpu_->getGuestNiceTimespan();

				for(unsigned int core = 0; core < cpu_->getCoreCount(); ++core) {
					startCoreUserTimespans_[core] = cpu_->getUserTimespan(core);
					startCoreNiceTimespans_[core] = cpu_->getNiceTimespan(core);
					startCoreSystemTimespans_[core] = cpu_->getSystemTimespan(core);
					startCoreIdleTimespans_[core] = cpu_->getIdleTimespan(core);
					startCoreIOWaitTimespans_[core] = cpu_->getIOWaitTimespan(core);
					startCoreInterruptsTimespans_[core] = cpu_->getInterruptsTimespan(core);
					startCoreSoftInterruptsTimespans_[core] = cpu_->getSoftInterruptsTimespan(core);
					startCoreStealTimespans_[core] = cpu_->getStealTimespan(core);
					startCoreGuestTimespans_[core] = cpu_->getGuestTimespan(core);
					startCoreGuestNiceTimespans_[core] = cpu_->getGuestNiceTimespan(core);
				}
			}

			auto cpuResults = std::map<std::string, std::string> {
				{ "userTimespan", std::to_string(cpu_->getUserTimespan() - startUserTimespan_) },
				{ "niceTimespan", std::to_string(cpu_->getNiceTimespan() - startNiceTimespan_) },
				{ "systemTimespan", std::to_string(cpu_->getSystemTimespan() - startSystemTimespan_) },
				{ "idleTimespan", std::to_string(cpu_->getIdleTimespan() - startIdleTimespan_) },
				{ "ioWaitTimespan", std::to_string(cpu_->getIOWaitTimespan() - startIOWaitTimespan_) },
				{ "interruptsTimespan", std::to_string(cpu_->getInterruptsTimespan() - startInterruptsTimespan_) },
				{ "softInterruptsTimespan", std::to_string(cpu_->getSoftInterruptsTimespan() - startSoftInterruptsTimespan_) },
				{ "stealTimespan", std::to_string(cpu_->getStealTimespan() - startStealTimespan_) },
				{ "guestTimespan", std::to_string(cpu_->getGuestTimespan() - startGuestTimespan_) },
				{ "guestNiceTimespan", std::to_string(cpu_->getGuestNiceTimespan() - startGuestNiceTimespan_) },
			};

			// Add the per-core values
			for(unsigned int core = 0; core < cpu_->getCoreCount(); ++core) {
				cpuResults["coreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getCoreClockRate(core));
				cpuResults["coreUtilizationRateCore" + std::to_string(core)] = std::to_string(cpu_->getCoreUtilizationRate(core));
				cpuResults["maximumCoreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getMaximumCoreClockRate(core));
				cpuResults["userTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getUserTimespan() - startCoreUserTimespans_[core]);
				cpuResults["niceTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getNiceTimespan() - startCoreNiceTimespans_[core]);
				cpuResults["systemTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getSystemTimespan() - startCoreSystemTimespans_[core]);
				cpuResults["idleTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getIdleTimespan() - startCoreIdleTimespans_[core]);
				cpuResults["ioWaitTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getIOWaitTimespan() - startCoreIOWaitTimespans_[core]);
				cpuResults["interruptsTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getInterruptsTimespan() - startCoreInterruptsTimespans_[core]);
				cpuResults["softInterruptsTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getSoftInterruptsTimespan() - startCoreSoftInterruptsTimespans_[core]);
				cpuResults["stealTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getStealTimespan() - startCoreStealTimespans_[core]);
				cpuResults["guestTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getGuestTimespan() - startCoreGuestTimespans_[core]);
				cpuResults["guestNiceTimespanCore" + std::to_string(core)] = std::to_string(cpu_->getGuestNiceTimespan() - startCoreGuestNiceTimespans_[core]);
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
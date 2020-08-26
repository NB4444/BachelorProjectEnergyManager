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
				{ "userTimespan", std::to_string((cpu_->getUserTimespan() - startUserTimespan_).count()) },
				{ "niceTimespan", std::to_string((cpu_->getNiceTimespan() - startNiceTimespan_).count()) },
				{ "systemTimespan", std::to_string((cpu_->getSystemTimespan() - startSystemTimespan_).count()) },
				{ "idleTimespan", std::to_string((cpu_->getIdleTimespan() - startIdleTimespan_).count()) },
				{ "ioWaitTimespan", std::to_string((cpu_->getIOWaitTimespan() - startIOWaitTimespan_).count()) },
				{ "interruptsTimespan", std::to_string((cpu_->getInterruptsTimespan() - startInterruptsTimespan_).count()) },
				{ "softInterruptsTimespan", std::to_string((cpu_->getSoftInterruptsTimespan() - startSoftInterruptsTimespan_).count()) },
				{ "stealTimespan", std::to_string((cpu_->getStealTimespan() - startStealTimespan_).count()) },
				{ "guestTimespan", std::to_string((cpu_->getGuestTimespan() - startGuestTimespan_).count()) },
				{ "guestNiceTimespan", std::to_string((cpu_->getGuestNiceTimespan() - startGuestNiceTimespan_).count()) },
			};

			// Add the per-core values
			for(unsigned int core = 0; core < cpu_->getCoreCount(); ++core) {
				cpuResults["coreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getCoreClockRate(core).toValue());
				cpuResults["coreUtilizationRateCore" + std::to_string(core)] = std::to_string(cpu_->getCoreUtilizationRate(core).toCombined());
				cpuResults["maximumCoreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getMaximumCoreClockRate(core).toValue());
				cpuResults["userTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getUserTimespan() - startCoreUserTimespans_[core]).count());
				cpuResults["niceTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getNiceTimespan() - startCoreNiceTimespans_[core]).count());
				cpuResults["systemTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getSystemTimespan() - startCoreSystemTimespans_[core]).count());
				cpuResults["idleTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getIdleTimespan() - startCoreIdleTimespans_[core]).count());
				cpuResults["ioWaitTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getIOWaitTimespan() - startCoreIOWaitTimespans_[core]).count());
				cpuResults["interruptsTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getInterruptsTimespan() - startCoreInterruptsTimespans_[core]).count());
				cpuResults["softInterruptsTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getSoftInterruptsTimespan() - startCoreSoftInterruptsTimespans_[core]).count());
				cpuResults["stealTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getStealTimespan() - startCoreStealTimespans_[core]).count());
				cpuResults["guestTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getGuestTimespan() - startCoreGuestTimespans_[core]).count());
				cpuResults["guestNiceTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getGuestNiceTimespan() - startCoreGuestNiceTimespans_[core]).count());
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
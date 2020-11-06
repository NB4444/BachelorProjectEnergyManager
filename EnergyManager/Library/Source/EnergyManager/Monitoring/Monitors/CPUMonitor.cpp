#include "./CPUMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> CPUMonitor::onPollProcessor() {
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

				std::map<std::string, std::string> results;
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["userTimespan"] = std::to_string((cpu_->getUserTimespan() - startUserTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["niceTimespan"] = std::to_string((cpu_->getNiceTimespan() - startNiceTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["systemTimespan"] = std::to_string((cpu_->getSystemTimespan() - startSystemTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["idleTimespan"] = std::to_string((cpu_->getIdleTimespan() - startIdleTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["ioWaitTimespan"] = std::to_string((cpu_->getIOWaitTimespan() - startIOWaitTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["interruptsTimespan"] = std::to_string((cpu_->getInterruptsTimespan() - startInterruptsTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["softInterruptsTimespan"] = std::to_string((cpu_->getSoftInterruptsTimespan() - startSoftInterruptsTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["stealTimespan"] = std::to_string((cpu_->getStealTimespan() - startStealTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["guestTimespan"] = std::to_string((cpu_->getGuestTimespan() - startGuestTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["guestNiceTimespan"] = std::to_string((cpu_->getGuestNiceTimespan() - startGuestNiceTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["turboEnabled"] = std::to_string(cpu_->getTurboEnabled()));

				// Add the per-core values
				for(unsigned int core = 0; core < cpu_->getCoreCount(); ++core) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["coreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getCoreClockRate(core).toValue()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["currentMaximumCoreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getCurrentMaximumCoreClockRate(core).toValue()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["currentMinimumCoreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getCurrentMinimumCoreClockRate(core).toValue()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["coreUtilizationRateCore" + std::to_string(core)] = std::to_string(cpu_->getCoreUtilizationRate(core).toCombined()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["maximumCoreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getMaximumCoreClockRate(core).toValue()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["minimumCoreClockRateCore" + std::to_string(core)] = std::to_string(cpu_->getMinimumCoreClockRate(core).toValue()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["userTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getUserTimespan(core) - startCoreUserTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["niceTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getNiceTimespan(core) - startCoreNiceTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["systemTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getSystemTimespan(core) - startCoreSystemTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["idleTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getIdleTimespan(core) - startCoreIdleTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["ioWaitTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getIOWaitTimespan(core) - startCoreIOWaitTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["interruptsTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getInterruptsTimespan(core) - startCoreInterruptsTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["softInterruptsTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getSoftInterruptsTimespan(core) - startCoreSoftInterruptsTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["stealTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getStealTimespan(core) - startCoreStealTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["guestTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getGuestTimespan(core) - startCoreGuestTimespans_[core]).count()));
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
						results["guestNiceTimespanCore" + std::to_string(core)] = std::to_string((cpu_->getGuestNiceTimespan(core) - startCoreGuestNiceTimespans_[core]).count()));
				}

				return results;
			}

			void CPUMonitor::onResetDevice() {
				startUserTimespan_ = {};
				startCoreUserTimespans_ = {};
				startNiceTimespan_ = {};
				startCoreNiceTimespans_ = {};
				startSystemTimespan_ = {};
				startCoreSystemTimespans_ = {};
				startIdleTimespan_ = {};
				startCoreIdleTimespans_ = {};
				startIOWaitTimespan_ = {};
				startCoreIOWaitTimespans_ = {};
				startInterruptsTimespan_ = {};
				startCoreInterruptsTimespans_ = {};
				startSoftInterruptsTimespan_ = {};
				startCoreSoftInterruptsTimespans_ = {};
				startStealTimespan_ = {};
				startCoreStealTimespans_ = {};
				startGuestTimespan_ = {};
				startCoreGuestTimespans_ = {};
				startGuestNiceTimespan_ = {};
				startCoreGuestNiceTimespans_ = {};
				startTimespansMeasured_ = false;
			}

			CPUMonitor::CPUMonitor(const std::shared_ptr<Hardware::CPU>& cpu, const std::chrono::system_clock::duration& interval) : ProcessorMonitor("CPUMonitor", cpu, interval), cpu_(cpu) {
				reset();
			}
		}
	}
}
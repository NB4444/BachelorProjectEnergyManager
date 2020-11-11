#include "./CentralProcessorMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> CentralProcessorMonitor::onPollProcessor() {
				if(!startTimespansMeasured_) {
					startTimespansMeasured_ = true;
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startUserTimespan_ = processor_->getUserTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startNiceTimespan_ = processor_->getNiceTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startSystemTimespan_ = processor_->getSystemTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startIdleTimespan_ = processor_->getIdleTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startIOWaitTimespan_ = processor_->getIOWaitTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startInterruptsTimespan_ = processor_->getInterruptsTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startSoftInterruptsTimespan_ = processor_->getSoftInterruptsTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startStealTimespan_ = processor_->getStealTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startGuestTimespan_ = processor_->getGuestTimespan());
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(startGuestNiceTimespan_ = processor_->getGuestNiceTimespan());
				}

				std::map<std::string, std::string> results;
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["userTimespan"] = std::to_string((processor_->getUserTimespan() - startUserTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["niceTimespan"] = std::to_string((processor_->getNiceTimespan() - startNiceTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["systemTimespan"] = std::to_string((processor_->getSystemTimespan() - startSystemTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["idleTimespan"] = std::to_string((processor_->getIdleTimespan() - startIdleTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["ioWaitTimespan"] = std::to_string((processor_->getIOWaitTimespan() - startIOWaitTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["interruptsTimespan"] = std::to_string((processor_->getInterruptsTimespan() - startInterruptsTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(
					results["softInterruptsTimespan"] = std::to_string((processor_->getSoftInterruptsTimespan() - startSoftInterruptsTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["stealTimespan"] = std::to_string((processor_->getStealTimespan() - startStealTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["guestTimespan"] = std::to_string((processor_->getGuestTimespan() - startGuestTimespan_).count()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["guestNiceTimespan"] = std::to_string((processor_->getGuestNiceTimespan() - startGuestNiceTimespan_).count()));

				// Get downstream values
				auto centralProcessorResults = onPollCentralProcessor();
				results.insert(centralProcessorResults.begin(), centralProcessorResults.end());

				return results;
			}

			std::map<std::string, std::string> CentralProcessorMonitor::onPollCentralProcessor() {
				return {};
			}

			void CentralProcessorMonitor::onResetDevice() {
				startUserTimespan_ = {};
				startNiceTimespan_ = {};
				startSystemTimespan_ = {};
				startIdleTimespan_ = {};
				startIOWaitTimespan_ = {};
				startInterruptsTimespan_ = {};
				startSoftInterruptsTimespan_ = {};
				startStealTimespan_ = {};
				startGuestTimespan_ = {};
				startGuestNiceTimespan_ = {};
				startTimespansMeasured_ = false;
			}

			CentralProcessorMonitor::CentralProcessorMonitor(const std::string& name, const std::shared_ptr<Hardware::CentralProcessor>& processor, const std::chrono::system_clock::duration& interval)
				: ProcessorMonitor(name, processor, interval)
				, processor_(processor) {
				reset();
			}
		}
	}
}
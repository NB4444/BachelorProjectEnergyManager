#include "./CentralProcessorMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			void CentralProcessorMonitor::beforeLoopStart() {
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

			std::map<std::string, std::string> CentralProcessorMonitor::onPollProcessor() {
				std::map<std::string, std::string> results;
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["userTimespan"] = Utility::Text::toString(processor_->getUserTimespan() - startUserTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["niceTimespan"] = Utility::Text::toString(processor_->getNiceTimespan() - startNiceTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["systemTimespan"] = Utility::Text::toString(processor_->getSystemTimespan() - startSystemTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["idleTimespan"] = Utility::Text::toString(processor_->getIdleTimespan() - startIdleTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["ioWaitTimespan"] = Utility::Text::toString(processor_->getIOWaitTimespan() - startIOWaitTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["interruptsTimespan"] = Utility::Text::toString(processor_->getInterruptsTimespan() - startInterruptsTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["softInterruptsTimespan"] = Utility::Text::toString(processor_->getSoftInterruptsTimespan() - startSoftInterruptsTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["stealTimespan"] = Utility::Text::toString(processor_->getStealTimespan() - startStealTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["guestTimespan"] = Utility::Text::toString(processor_->getGuestTimespan() - startGuestTimespan_));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["guestNiceTimespan"] = Utility::Text::toString(processor_->getGuestNiceTimespan() - startGuestNiceTimespan_));

				// Get downstream values
				auto centralProcessorResults = onPollCentralProcessor();
				results.insert(centralProcessorResults.begin(), centralProcessorResults.end());

				return results;
			}

			std::map<std::string, std::string> CentralProcessorMonitor::onPollCentralProcessor() {
				return {};
			}

			CentralProcessorMonitor::CentralProcessorMonitor(const std::string& name, const std::shared_ptr<Hardware::CentralProcessor>& processor, const std::chrono::system_clock::duration& interval)
				: ProcessorMonitor(name, processor, interval)
				, processor_(processor) {
			}
		}
	}
}
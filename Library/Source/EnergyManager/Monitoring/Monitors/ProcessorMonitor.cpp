#include "./ProcessorMonitor.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Monitoring {
		namespace Monitors {
			std::map<std::string, std::string> ProcessorMonitor::onPollDevice() {
				std::map<std::string, std::string> results;
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["id"] = Utility::Text::toString(processor_->getID()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["coreClockRate"] = Utility::Text::toString(processor_->getCoreClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["coreUtilizationRate"] = Utility::Text::toString(processor_->getCoreUtilizationRate().toCombined()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["currentMaximumCoreClockRate"] = Utility::Text::toString(processor_->getCurrentMaximumCoreClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["currentMinimumCoreClockRate"] = Utility::Text::toString(processor_->getCurrentMinimumCoreClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["maximumCoreClockRate"] = Utility::Text::toString(processor_->getMaximumCoreClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["minimumCoreClockRate"] = Utility::Text::toString(processor_->getMinimumCoreClockRate().toValue()));
				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(results["temperature"] = Utility::Text::toString(processor_->getTemperature().toValue()));

				// Get downstream values
				auto processorResults = onPollProcessor();
				results.insert(processorResults.begin(), processorResults.end());

				return results;
			}

			std::map<std::string, std::string> ProcessorMonitor::onPollProcessor() {
				return {};
			}

			ProcessorMonitor::ProcessorMonitor(const std::string& name, const std::shared_ptr<Hardware::Processor>& processor, const std::chrono::system_clock::duration& interval)
				: DeviceMonitor(name, processor, interval)
				, processor_(processor) {
			}
		}
	}
}
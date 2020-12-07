#include "./CPU.hpp"

#include "EnergyManager/Hardware/Core.hpp"
#include "EnergyManager/Utility/Collections.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <fstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

/**
 * More performance variables to monitor can be found at these sources:
 * | Tool              | Functionality                      | URL                                                                                                         |
 * | :---------------- | :--------------------------------- | :---------------------------------------------------------------------------------------------------------- |
 * | General           | Retrieving memory / CPU parameters | https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process |
 * | Intel RAPL        | Information                        | https://developer.mozilla.org/en-US/docs/Mozilla/Performance/tools_power_rapl                               |
 * | Intel RAPL        | General usage                      | http://web.eece.maine.edu/~vweaver/projects/rapl/                                                           |
 * | Intel RAPL        | Model                              | https://blog.chih.me/read-cpu-power-with-RAPL.html                                                          |
 * | LIKWID Powermeter | Information                        | https://github.com/RRZE-HPC/likwid/wiki/Likwid-Powermeter                                                   |
 * | PAPI              |                                    |                                                                                                             |
 */

namespace EnergyManager {
	namespace Hardware {
		CPU::CPU(const unsigned int& id) : CentralProcessor(id), powerConsumption(std::chrono::milliseconds(10)) {
			// Set initial values
			startEnergyConsumption_ = getEnergyConsumption();
			lastEnergyConsumption_ = startEnergyConsumption_;

			// Detect and add cores
			for(unsigned int coreID = 0; coreID < getProcCPUInfoValuesPerCPU().at(id).size(); ++coreID) {
				cores_.push_back(Core::getCore(this, coreID));
			}
		}

		std::vector<std::string> CPU::generateHeaders() const {
			return { "CPU " + Utility::Text::toString(getID()) };
		}

		std::vector<std::shared_ptr<Hardware::CPU>> CPU::parseCPUs(const std::string& cpuString) {
			std::vector<std::string> cpuStrings = EnergyManager::Utility::Text::splitToVector(cpuString, ",", true);
			std::vector<std::shared_ptr<EnergyManager::Hardware::CPU>> cpus;
			std::transform(cpuStrings.begin(), cpuStrings.end(), std::back_inserter(cpus), [](const auto& cpuString) {
				return EnergyManager::Hardware::CPU::getCPU(std::stoi(cpuString));
			});

			return cpus;
		}

		std::shared_ptr<CPU> CPU::getCPU(const unsigned int& id) {
			// Only allow one thread to get CPUs at a time
			static std::mutex mutex;
			std::lock_guard<std::mutex> guard(mutex);

			// Keep track of CPUs
			static std::map<uint32_t, std::shared_ptr<CPU>> cpus = {};

			auto iterator = cpus.find(id);
			if(iterator == cpus.end()) {
				cpus[id] = std::shared_ptr<CPU>(new CPU(id));
			}

			return cpus[id];
		}

		std::vector<std::shared_ptr<CPU>> CPU::getCPUs() {
			std::vector<std::shared_ptr<CPU>> cpus = {};
			for(unsigned int cpu = 0; cpu < getCPUCount(); ++cpu) {
				cpus.push_back(getCPU(cpu));
			}

			return cpus;
		}

		unsigned int CPU::getCPUCount() {
			return getProcCPUInfoValuesPerCPU().size();
		}

		std::vector<std::shared_ptr<Core>> CPU::getCores() const {
			return cores_;
		}

		Utility::Units::Hertz CPU::getCoreClockRate() const {
			Utility::Units::Hertz sum = 0;

			for(const auto& core : getCores()) {
				sum += core->getCoreClockRate();
			}

			return sum / getCores().size();
		}

		Utility::Units::Hertz CPU::getCurrentMinimumCoreClockRate() const {
			Utility::Units::Hertz minimum = -1;

			for(const auto& core : getCores()) {
				auto current = core->getCurrentMinimumCoreClockRate();
				if(minimum < 0 || current < minimum) {
					minimum = current;
				}
			}

			return minimum;
		}

		Utility::Units::Hertz CPU::getCurrentMaximumCoreClockRate() const {
			Utility::Units::Hertz maximum = 0;

			for(const auto& core : getCores()) {
				auto current = core->getCurrentMinimumCoreClockRate();
				if(current > maximum) {
					maximum = current;
				}
			}

			return maximum;
		}

		void CPU::setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) {
			logDebug("Setting clock rate range to [%lu, %lu]...", minimumRate.toValue(), minimumRate.toValue());

			for(const auto& core : getCores()) {
				core->setCoreClockRate(minimumRate, maximumRate);
			}

			// Check the type of power scaling driver in use
			if(getCores()[0]->getPowerScalingDriver() == "intel_pstate") {
				// Set minimum rate
				std::ofstream minimumRateStream("/sys/devices/system/cpu/intel_pstate/min_perf_pct");
				const auto minimumRatePercentage = static_cast<unsigned int>((static_cast<double>(minimumRate.toValue()) / static_cast<double>(getMaximumCoreClockRate().toValue())) * 100);
				minimumRateStream << minimumRatePercentage;

				// Set maximum rate
				std::ofstream maximumRateStream("/sys/devices/system/cpu/intel_pstate/max_perf_pct");
				const auto maximumRatePercentage = static_cast<unsigned int>((static_cast<double>(maximumRate.toValue()) / static_cast<double>(getMaximumCoreClockRate().toValue())) * 100);
				maximumRateStream << maximumRatePercentage;
			}
		}

		void CPU::resetCoreClockRate() {
			logDebug("Resetting clock rate...");

			for(const auto& core : getCores()) {
				core->resetCoreClockRate();
			}

			// Check the type of power scaling driver in use
			if(getCores()[0]->getPowerScalingDriver() == "intel_pstate") {
				// Set minimum rate
				std::ofstream minimumRateStream("/sys/devices/system/cpu/intel_pstate/min_perf_pct");
				minimumRateStream << 0;

				// Set maximum rate
				std::ofstream maximumRateStream("/sys/devices/system/cpu/intel_pstate/max_perf_pct");
				maximumRateStream << 100;
			}
		}

		Utility::Units::Percent CPU::getCoreUtilizationRate() {
			double sum = 0;

			for(const auto& core : getCores()) {
				sum += core->getCoreUtilizationRate().getUnit();
			}

			return sum / getCores().size();
		}

		Utility::Units::Joule CPU::getEnergyConsumption() {
			return Utility::Units::Joule(
					   std::stod(Utility::Text::readFile("/sys/class/powercap/intel-rapl/intel-rapl:" + Utility::Text::toString(getID()) + "/energy_uj")),
					   Utility::Units::SIPrefix::MICRO)
				   - startEnergyConsumption_;
		}

		Utility::Units::Hertz CPU::getMinimumCoreClockRate() const {
			Utility::Units::Hertz minimum = 0;
			bool found = false;

			for(const auto& core : getCores()) {
				auto currentCoreClockRate = core->getMinimumCoreClockRate();
				if(!found || currentCoreClockRate < minimum) {
					minimum = currentCoreClockRate;
					found = true;
				}
			}

			return minimum;
		}

		Utility::Units::Hertz CPU::getMaximumCoreClockRate() const {
			Utility::Units::Hertz maximum = 0;

			for(const auto& core : getCores()) {
				auto currentCoreClockRate = core->getMaximumCoreClockRate();
				if(currentCoreClockRate > maximum) {
					maximum = currentCoreClockRate;
				}
			}

			return maximum;
		}

		Utility::Units::Watt CPU::getPowerConsumption() {
			return powerConsumption.getValue([&](const auto& value, const auto& timeSinceLastUpdate) {
				// Get the time since last poll in seconds, with decimals
				const auto pollingTimespan = std::chrono::duration<double>(timeSinceLastUpdate).count();

				// Keep the value
				Utility::Units::Watt powerConsumption = value;

				// Get the current values
				const auto currentEnergyConsumption = getEnergyConsumption();

				// Calculate the power consumption in Watts
				if(pollingTimespan == 0) {
					logWarning("Polling timespan equal to zero, can't measure power consumption");
				} else {
					// Get the difference
					const auto energyConsumed = currentEnergyConsumption - lastEnergyConsumption_;

					// FIXME: There can be some noise in the data so we filter out values that are too high
					powerConsumption = Utility::Units::Watt(energyConsumed.toValue() / pollingTimespan);
					if(powerConsumption.toValue() <= 0 || powerConsumption > 300) {
						logWarning(
							"Detected unusual power consumption of %f based on:\n"
							"Current energy consumption: %f\n"
							"Last energy consumption: %f\n"
							"Energy consumed: %f\n"
							"Timespan: %s",
							powerConsumption.toValue(),
							currentEnergyConsumption.toValue(),
							lastEnergyConsumption_.toValue(),
							energyConsumed.toValue(),
							Utility::Text::formatDuration(timeSinceLastUpdate).c_str());
					}
					if(powerConsumption.toValue() < 0) {
						powerConsumption = 0;
					}
				}

				lastEnergyConsumption_ = currentEnergyConsumption;

				return powerConsumption;
			});
		}

		std::chrono::system_clock::duration CPU::getUserTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getUserTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getNiceTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getNiceTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getSystemTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getSystemTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getIdleTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getIdleTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getIOWaitTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getIOWaitTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getInterruptsTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getInterruptsTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getSoftInterruptsTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getSoftInterruptsTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getStealTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getStealTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getGuestTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getGuestTimespan();
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getGuestNiceTimespan() const {
			std::chrono::system_clock::duration sum;

			for(const auto& core : getCores()) {
				sum += core->getGuestNiceTimespan();
			}

			return sum;
		}

		Utility::Units::Celsius CPU::getTemperature() const {
			std::ifstream inputStream("/sys/class/thermal/thermal_zone0/temp");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Celsius(std::stoi(cpuInfo), Utility::Units::SIPrefix::MILLI);
		}

		bool CPU::getTurboEnabled() const {
			std::ifstream inputStream("/sys/devices/system/cpu/intel_pstate/no_turbo");
			std::string turboEnabled((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return std::stoi(turboEnabled);
		}

		void CPU::setTurboEnabled(const bool& turbo) const {
			logDebug("Setting turbo enabled to %d...", turbo);

			std::ofstream outputStream("/sys/devices/system/cpu/intel_pstate/no_turbo");
			outputStream << !turbo;
		}
	}
}
#include "./CPU.hpp"

#include "EnergyManager/Utility/Text.hpp"

#include <fstream>
#include <string>
#include <unistd.h>
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
		std::map<unsigned int, std::shared_ptr<CPU>> CPU::cpus_;

		std::map<unsigned int, std::map<std::string, std::string>> CPU::getProcCPUInfoValues() {
			// Read the CPU info
			std::ifstream inputStream("/proc/cpuinfo");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			// Parse the values to lines
			std::vector<std::string> cpuInfoLines = Utility::Text::split(cpuInfo, "\n");

			// Parse the values per CPU
			std::map<unsigned int, std::map<std::string, std::string>> result;
			unsigned int currentCPUID;
			for(const auto& cpuInfoLine : cpuInfoLines) {
				// Parse the current value
				std::vector<std::string> valuePair = Utility::Text::split(cpuInfoLine, ":");
				std::transform(valuePair.begin(), valuePair.end(), valuePair.begin(), [](const std::string& value) {
					return Utility::Text::trim(value);
				});

				// Do something based on the value type
				if(valuePair.front() == "processor") {
					// Set the current CPU ID
					currentCPUID = std::stoi(valuePair.back());
				} else {
					// Set the variable
					result[currentCPUID][valuePair.front()] = valuePair.back();
				}
			}

			return result;
		}

		CPU::CPU(const unsigned int& id)
			: id_(id)
			, monitorThread_([&] {
				while(monitorThreadRunning_) {
					// Measure starting energy consumption
					if(!startEnergyConsumptionMeasured_) {
						startEnergyConsumptionMeasured_ = true;

						startEnergyConsumption_ = getEnergyConsumption();
					}

					// Get the current values
					auto currentEnergyConsumption = getEnergyConsumption();
					auto currentTimestamp = std::chrono::system_clock::now();
					auto pollingTimespan = std::chrono::duration_cast<std::chrono::seconds>(currentTimestamp - lastEnergyConsumptionPollTimestamp_).count();

					// Calculate the power consumption in Watts
					powerConsumption_ = (currentEnergyConsumption - lastEnergyConsumption_) / (pollingTimespan == 0 ? 0.1 : pollingTimespan);

					// Set the variables for the next poll cycle
					lastEnergyConsumption_ = currentEnergyConsumption;
					lastEnergyConsumptionPollTimestamp_ = currentTimestamp;

					usleep(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::milliseconds (100)).count());
				}
			}) {
		}

		std::shared_ptr<CPU> CPU::getCPU(const unsigned int& id) {
			auto iterator = cpus_.find(id);
			if(iterator == cpus_.end()) {
				cpus_[id] = std::shared_ptr<CPU>(new CPU(id));
			}

			return cpus_[id];
		}

		CPU::~CPU() {
			// Stop the monitor
			monitorThreadRunning_ = false;
			monitorThread_.join();
		}

		unsigned long CPU::getCoreClockRate() const {
			return std::stof(getProcCPUInfoValues()[id_]["cpu MHz"]) * 1000000l;
		}

		void CPU::setCoreClockRate(unsigned long& rate) {
			// TODO
		}

		float CPU::getEnergyConsumption() const {
			if(startEnergyConsumptionMeasured_) {
				std::ifstream inputStream("/sys/class/powercap/intel-rapl/intel-rapl:" + std::to_string(id_) + "/energy_uj");
				std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

				return (std::stol(cpuInfo) / 1e6l) - startEnergyConsumption_;
			} else {
				return 0;
			}
		}

		unsigned long CPU::getMaximumCoreClockRate() const {
			std::ifstream inputStream("/sys/devices/system/cpu/cpu" + std::to_string(id_) + "/cpufreq/cpuinfo_max_freq");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return std::stoi(cpuInfo) * 1000l;
		}

		float CPU::getPowerConsumption() const {
			return powerConsumption_;
		}
	}
}
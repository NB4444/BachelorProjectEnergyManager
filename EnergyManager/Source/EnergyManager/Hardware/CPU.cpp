#include "./CPU.hpp"

#include "EnergyManager/Utility/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <fstream>
#include <string>
#include <sys/param.h>
#include <sys/types.h>
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

		std::mutex CPU::monitorThreadMutex_;

		std::map<unsigned int, std::map<std::string, std::string>> CPU::getProcCPUInfoValuesPerProcessor() {
			// Read the CPU info
			std::ifstream inputStream("/proc/cpuinfo");
			std::string processorInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			// Parse the values to lines
			std::vector<std::string> processorInfoLines = Utility::Text::splitToVector(processorInfo, "\n");

			// Parse the values per core
			std::map<unsigned int, std::map<std::string, std::string>> processorValues;
			unsigned int currentProcessorID;
			for(const auto& processorInfoLine : processorInfoLines) {
				// Parse the current value
				std::vector<std::string> valuePair = Utility::Text::splitToVector(processorInfoLine, ":");
				std::transform(valuePair.begin(), valuePair.end(), valuePair.begin(), [](const std::string& value) {
					return Utility::Text::trim(value);
				});

				// Do something based on the value type
				if(valuePair.front() == "processor") {
					// Set the current CPU ID
					currentProcessorID = std::stoi(valuePair.back());
				} else {
					// Set the variable
					processorValues[currentProcessorID][valuePair.front()] = valuePair.back();
				}
			}

			return processorValues;
		}

		std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> CPU::getProcCPUInfoValuesPerCPU() {
			// Parse the values per CPU
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> cpuCoreValues;
			for(auto& processorValues : getProcCPUInfoValuesPerProcessor()) {
				auto cpuID = std::stoi(processorValues.second["physical id"]);

				// Create structures if they don't exist
				if(cpuCoreValues.find(cpuID) == cpuCoreValues.end()) {
					cpuCoreValues[cpuID] = {};
				}

				auto coreID = std::stoi(processorValues.second["core id"]);
				cpuCoreValues[cpuID][coreID] = processorValues.second;
			}

			return cpuCoreValues;
		}

		std::map<unsigned int, std::map<std::string, double>> CPU::getProcStatValuesPerProcessor() {
			// Read the CPU info
			std::ifstream inputStream("/proc/stat");
			std::string processorInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			// Parse the values to lines
			std::vector<std::string> processorInfoLines = Utility::Text::splitToVector(processorInfo, "\n");

			// Parse the lines
			std::map<unsigned int, std::map<std::string, double>> processorValues;
			for(const auto& processorInfoLine : processorInfoLines) {
				// Only process processors
				if(processorInfoLine.rfind("cpu", 0) != 0) {
					break;
				}

				// Get the values
				std::vector<std::string> processorInfoValues = Utility::Text::splitToVector(Utility::Text::mergeWhitespace(processorInfoLine), " ");
				std::string processorName = processorInfoValues[0];

				// Skip processors without ID
				if(processorName.size() == 3) {
					continue;
				}

				unsigned int processorID = std::stoi(processorName.substr(3, processorName.size() - 3));

				// Get Jiffies per seconds to convert the values
				const double jiffiesPerSecond = sysconf(_SC_CLK_TCK);

				processorValues[processorID]["userTimespan"] = std::stol(processorInfoValues[1]) / jiffiesPerSecond;
				processorValues[processorID]["niceTimespan"] = std::stol(processorInfoValues[2]) / jiffiesPerSecond;
				processorValues[processorID]["systemTimespan"] = std::stol(processorInfoValues[3]) / jiffiesPerSecond;
				processorValues[processorID]["idleTimespan"] = std::stol(processorInfoValues[4]) / jiffiesPerSecond;
				processorValues[processorID]["ioWaitTimespan"] = std::stol(processorInfoValues[5]) / jiffiesPerSecond;
				processorValues[processorID]["interruptsTimespan"] = std::stol(processorInfoValues[6]) / jiffiesPerSecond;
				processorValues[processorID]["softInterruptsTimespan"] = std::stol(processorInfoValues[7]) / jiffiesPerSecond;
				processorValues[processorID]["stealTimespan"] = std::stol(processorInfoValues[8]) / jiffiesPerSecond;
				processorValues[processorID]["guestTimespan"] = std::stol(processorInfoValues[9]) / jiffiesPerSecond;
				processorValues[processorID]["guestNiceTimespan"] = std::stol(processorInfoValues[10]) / jiffiesPerSecond;
			}

			return processorValues;
		}

		std::map<unsigned int, std::map<unsigned int, std::map<std::string, double>>> CPU::getProcStatValuesPerCPU() {
			auto procCPUInfoValuesPerProcessor = getProcCPUInfoValuesPerProcessor();

			// Parse the values per CPU
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, double>>> cpuCoreValues;
			for(auto& processorValues : getProcStatValuesPerProcessor()) {
				auto processorID = processorValues.first;

				auto cpuID = std::stoi(procCPUInfoValuesPerProcessor[processorID]["physical id"]);

				// Create structures if they don't exist
				if(cpuCoreValues.find(cpuID) == cpuCoreValues.end()) {
					cpuCoreValues[cpuID] = {};
				}

				auto coreID = std::stoi(procCPUInfoValuesPerProcessor[processorID]["core id"]);
				cpuCoreValues[cpuID][coreID] = processorValues.second;
			}

			return cpuCoreValues;
		}

		CPU::CPU(const unsigned int& id) : id_(id) {
			monitorThread_ = std::thread([&] {
				while(monitorThreadRunning_) {
					{
						std::lock_guard<std::mutex> guard(monitorThreadMutex_);

						// Get the current values
						auto currentProcStatValues = getProcStatValuesPerCPU();
						auto currentEnergyConsumption = getEnergyConsumption();
						auto currentTimestamp = std::chrono::system_clock::now();
						auto pollingTimespan = std::chrono::duration_cast<std::chrono::seconds>(currentTimestamp - lastMonitorTimestamp_).count();

						// Calculate the power consumption in Watts
						auto divisor = (pollingTimespan == 0
							? 0.1
							: pollingTimespan);
						powerConsumption_ = (currentEnergyConsumption - lastEnergyConsumption_) / divisor;

						// Calculate the core utilization rates
						for(unsigned int core = 0; core < getCoreCount(); ++core) {
							auto previousIdle = lastProcStatValues_[id_][core]["idleTimespan"] + lastProcStatValues_[id_][core]["ioWaitTimespan"];
							auto previousActive = lastProcStatValues_[id][core]["userTimespan"] + lastProcStatValues_[id_][core]["niceTimespan"] + lastProcStatValues_[id_][core]["systemTimespan"]
								+ lastProcStatValues_[id_][core]["interruptsTimespan"] + lastProcStatValues_[id_][core]["softInterruptsTimespan"]
								+ lastProcStatValues_[id_][core]["stealTimespan"] + lastProcStatValues_[id_][core]["guestTimespan"]
								+ lastProcStatValues_[id_][core]["guestNiceTimespan"];
							auto previousTotal = previousIdle + previousActive;

							auto idle = currentProcStatValues[id_][core]["idleTimespan"] + currentProcStatValues[id_][core]["ioWaitTimespan"];
							auto active = currentProcStatValues[id][core]["userTimespan"] + currentProcStatValues[id_][core]["niceTimespan"] + currentProcStatValues[id_][core]["systemTimespan"]
								+ currentProcStatValues[id_][core]["interruptsTimespan"] + currentProcStatValues[id_][core]["softInterruptsTimespan"]
								+ currentProcStatValues[id_][core]["stealTimespan"] + currentProcStatValues[id_][core]["guestTimespan"]
								+ currentProcStatValues[id_][core]["guestNiceTimespan"];
							auto total = idle + active;

							auto totalDifference = total - previousTotal;
							auto idleDifference = idle - previousIdle;

							coreUtilizationRates_[core] = totalDifference == 0
								? 0
								: (static_cast<float>(totalDifference - idleDifference) / static_cast<float>(totalDifference) * 100);
						}

						// Set the variables for the next poll cycle
						lastProcCPUInfoValues_ = getProcCPUInfoValuesPerCPU();
						lastProcStatValues_ = currentProcStatValues;
						lastEnergyConsumption_ = currentEnergyConsumption;
						lastMonitorTimestamp_ = currentTimestamp;
					}

					usleep(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::milliseconds(100)).count());
				}
			});

			startEnergyConsumption_ = getEnergyConsumption();
		}

		unsigned int CPU::getProcessorID(const unsigned int& core) const {
			for(auto& procCpuInfoValuesPerProcessor : getProcCPUInfoValuesPerProcessor()) {
				if(std::stoi(procCpuInfoValuesPerProcessor.second["physical id"]) == id_ && std::stoi(procCpuInfoValuesPerProcessor.second["core id"]) == core) {
					return procCpuInfoValuesPerProcessor.first;
				}
			}

			ENERGY_MANAGER_UTILITY_EXCEPTION("Cannot find core");
		}

		double CPU::getProcStatTimespan(const unsigned int& core, const std::string& name) const {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			auto lastValue = lastProcStatValues_.at(id_).at(core).at(name);
			auto startValue = startProcStatValues_.at(id_).at(core).at(name);

			return lastValue - startValue;
		}

		std::shared_ptr<CPU> CPU::getCPU(const unsigned int& id) {
			auto iterator = cpus_.find(id);
			if(iterator == cpus_.end()) {
				cpus_[id] = std::shared_ptr<CPU>(new CPU(id));
			}

			return cpus_[id];
		}

		unsigned int CPU::getCPUCount() {
			return getProcCPUInfoValuesPerCPU().size();
		}

		CPU::~CPU() {
			// Stop the monitor
			monitorThreadRunning_ = false;
			monitorThread_.join();
		}

		unsigned long CPU::getCoreClockRate() const {
			unsigned long sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getCoreClockRate(coreIndex);
			}

			return sum / getCoreCount();
		}

		unsigned long CPU::getCoreClockRate(const unsigned int& core) const {
			return std::stof(lastProcCPUInfoValues_.at(id_).at(core).at("cpu MHz")) * 1000000l;
		}

		void CPU::setCoreClockRate(unsigned long& rate) {
			// TODO
			ENERGY_MANAGER_UTILITY_EXCEPTION("Not implemented");
		}

		float CPU::getCoreUtilizationRate() const {
			float sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getCoreUtilizationRate(coreIndex);
			}

			return sum / getCoreCount();
		}

		float CPU::getCoreUtilizationRate(const unsigned int& core) const {
			return coreUtilizationRates_.at(core);
		}

		float CPU::getEnergyConsumption() const {
			std::ifstream inputStream("/sys/class/powercap/intel-rapl/intel-rapl:" + std::to_string(id_) + "/energy_uj");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return (std::stol(cpuInfo) / 1e6l) - startEnergyConsumption_;
		}

		unsigned long CPU::getMaximumCoreClockRate() const {
			unsigned long maximum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				auto currentCoreClockRate = getCoreClockRate(coreIndex);
				if(currentCoreClockRate > maximum) {
					maximum = currentCoreClockRate;
				}
			}

			return maximum;
		}

		unsigned long CPU::getMaximumCoreClockRate(const unsigned int& core) const {
			std::ifstream inputStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/cpuinfo_max_freq");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return std::stoi(cpuInfo) * 1000l;
		}

		float CPU::getPowerConsumption() const {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			return powerConsumption_;
		}

		unsigned int CPU::getCoreCount() const {
			return getProcCPUInfoValuesPerCPU()[id_].size();
		}

		double CPU::getUserTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getUserTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getUserTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "userTimespan");
		}

		double CPU::getNiceTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getNiceTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getNiceTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "niceTimespan");
		}

		double CPU::getSystemTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getSystemTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getSystemTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "systemTimespan");
		}

		double CPU::getIdleTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getIdleTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getIdleTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "idleTimespan");
		}

		double CPU::getIOWaitTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getIOWaitTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getIOWaitTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "ioWaitTimespan");
		}

		double CPU::getInterruptsTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getInterruptsTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getInterruptsTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "interruptsTimespan");
		}

		double CPU::getSoftInterruptsTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getSoftInterruptsTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getSoftInterruptsTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "softInterruptsTimespan");
		}

		double CPU::getStealTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getStealTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getStealTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "stealTimespan");
		}

		double CPU::getGuestTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getGuestTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getGuestTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "guestTimespan");
		}

		double CPU::getGuestNiceTimespan() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getGuestNiceTimespan(coreIndex);
			}

			return sum;
		}

		double CPU::getGuestNiceTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "guestNiceTimespan");
		}
	}
}
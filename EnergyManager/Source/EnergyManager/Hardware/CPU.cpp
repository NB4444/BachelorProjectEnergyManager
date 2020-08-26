#include "./CPU.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
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

		std::chrono::system_clock::time_point CPU::lastProcCPUInfoValuesPerProcessorRetrieval = std::chrono::system_clock::now();

		std::map<unsigned int, std::map<std::string, std::string>> CPU::procCPUInfoValues = {};

		std::mutex CPU::procCPUInfoValuesMutex_;

		std::chrono::system_clock::time_point CPU::lastProcStatValuesRetrieval = std::chrono::system_clock::now();

		std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> CPU::procStatValues = {};

		std::mutex CPU::procStatValuesMutex_;

		std::mutex CPU::monitorThreadMutex_;

		std::map<unsigned int, std::map<std::string, std::string>> CPU::getProcCPUInfoValuesPerProcessor() {
			std::lock_guard<std::mutex> guard(procCPUInfoValuesMutex_);

			if(procCPUInfoValues.empty() || std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastProcCPUInfoValuesPerProcessorRetrieval).count() > 100) {
				procCPUInfoValues.clear();
				lastProcCPUInfoValuesPerProcessorRetrieval = std::chrono::system_clock::now();

				// Read the CPU info
				std::ifstream inputStream("/proc/cpuinfo");
				std::string processorInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

				// Parse the values to lines
				std::vector<std::string> processorInfoLines = Utility::Text::splitToVector(processorInfo, "\n");

				// Parse the values per core
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
						procCPUInfoValues[currentProcessorID][valuePair.front()] = valuePair.back();
					}
				}
			}

			return procCPUInfoValues;
		}

		std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> CPU::getProcCPUInfoValuesPerCPU() {
			// Parse the values per CPU
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> cpuCoreValues = {};
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

		std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> CPU::getProcStatValuesPerProcessor() {
			std::lock_guard<std::mutex> guard(procStatValuesMutex_);

			if(procStatValues.empty() || std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastProcStatValuesRetrieval).count() > 100) {
				procStatValues.clear();
				lastProcStatValuesRetrieval = std::chrono::system_clock::now();

				// Read the CPU info
				std::ifstream inputStream("/proc/stat");
				std::string processorInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

				// Parse the values to lines
				std::vector<std::string> processorInfoLines = Utility::Text::splitToVector(processorInfo, "\n");

				// Parse the lines
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

					auto setValue = [&](const std::string& name, const size_t& valueIndex) {
						// Get Jiffies per seconds to convert the values
						const double jiffiesPerSecond = sysconf(_SC_CLK_TCK);
						const double jiffiesPerMillisecond = jiffiesPerSecond / 1e3;

						procStatValues[processorID][name] = std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(static_cast<unsigned long>(std::stol(processorInfoValues[valueIndex]) / jiffiesPerMillisecond)));
					};

					setValue("userTimespan", 1);
					setValue("niceTimespan", 2);
					setValue("systemTimespan", 3);
					setValue("idleTimespan", 4);
					setValue("ioWaitTimespan", 5);
					setValue("interruptsTimespan", 6);
					setValue("softInterruptsTimespan", 7);
					setValue("stealTimespan", 8);
					setValue("guestTimespan", 9);
					setValue("guestNiceTimespan", 10);
				}
			}

			return procStatValues;
		}

		std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> CPU::getProcStatValuesPerCPU() {
			auto procCPUInfoValuesPerProcessor = getProcCPUInfoValuesPerProcessor();

			// Parse the values per CPU
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> cpuCoreValues;
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
						auto pollingTimespan = std::chrono::duration_cast<std::chrono::milliseconds>(currentTimestamp - lastMonitorTimestamp_).count() / static_cast<float>(1000);

						// Calculate the power consumption in Watts
						auto divisor = (pollingTimespan == 0
							? 0.1
							: pollingTimespan);
						powerConsumption_ = (currentEnergyConsumption - lastEnergyConsumption_).toValue() / divisor;

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
							auto activeDifference = active - previousActive;

							coreUtilizationRates_[core] = totalDifference.count() == 0
								? 0
								: (static_cast<double>(activeDifference.count()) / static_cast<double>(totalDifference.count()) * 100);
						}

						// Set the variables for the next poll cycle
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

			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Cannot find core");
		}

		std::chrono::system_clock::duration CPU::getProcStatTimespan(const unsigned int& core, const std::string& name) const {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			auto lastValue = getProcStatValuesPerCPU()[id_][core][name];
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

		Utility::Units::Hertz CPU::getCoreClockRate() const {
			Utility::Units::Hertz sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getCoreClockRate(coreIndex);
			}

			return sum / getCoreCount();
		}

		Utility::Units::Hertz CPU::getCoreClockRate(const unsigned int& core) const {
			std::ifstream coreClockRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/scaling_cur_freq");
			std::string coreClockRateString((std::istreambuf_iterator<char>(coreClockRateStream)), std::istreambuf_iterator<char>());

			// FIXME: Something may not be right with this output (see visualization results)
			return Utility::Units::Hertz(std::stoul(coreClockRateString), Utility::Units::SIPrefix::KILO);
		}

		void CPU::setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maxiumRate) {
			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				setCoreClockRate(coreIndex, minimumRate, maxiumRate);
			}
		}

		void CPU::setCoreClockRate(const unsigned int& core, const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) {
			// Set minimum rate
			std::ofstream minimumRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/scaling_min_freq");
			minimumRateStream << minimumRate.convertPrefix(Utility::Units::SIPrefix::KILO);

			// Set maximum rate
			std::ofstream maximumRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/scaling_max_freq");
			maximumRateStream << maximumRate.convertPrefix(Utility::Units::SIPrefix::KILO);
		}

		void CPU::resetCoreClockRate() {
			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				resetCoreClockRate(coreIndex);
			}
		}

		void CPU::resetCoreClockRate(const unsigned int& core) {
			std::ifstream minimumRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/cpuinfo_min_freq");
			std::string minimumRateString((std::istreambuf_iterator<char>(minimumRateStream)), std::istreambuf_iterator<char>());
			Utility::Units::Hertz minimumRate(std::stoul(minimumRateString), Utility::Units::SIPrefix::KILO);

			std::ifstream maximumRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/cpuinfo_max_freq");
			std::string maximumRateString((std::istreambuf_iterator<char>(maximumRateStream)), std::istreambuf_iterator<char>());
			Utility::Units::Hertz maximumRate(std::stoul(maximumRateString), Utility::Units::SIPrefix::KILO);

			setCoreClockRate(core, minimumRate, maximumRate);
		}

		Utility::Units::Percent CPU::getCoreUtilizationRate() const {
			double sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getCoreUtilizationRate(coreIndex).getUnit();
			}

			return sum / getCoreCount();
		}

		Utility::Units::Percent CPU::getCoreUtilizationRate(const unsigned int& core) const {
			return coreUtilizationRates_.at(core);
		}

		Utility::Units::Joule CPU::getEnergyConsumption() const {
			std::ifstream inputStream("/sys/class/powercap/intel-rapl/intel-rapl:" + std::to_string(id_) + "/energy_uj");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Joule(std::stol(cpuInfo), Utility::Units::SIPrefix::MICRO) - startEnergyConsumption_;
		}

		Utility::Units::Hertz CPU::getMaximumCoreClockRate() const {
			Utility::Units::Hertz maximum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				auto currentCoreClockRate = getCoreClockRate(coreIndex);
				if(currentCoreClockRate > maximum) {
					maximum = currentCoreClockRate;
				}
			}

			return maximum;
		}

		Utility::Units::Hertz CPU::getMaximumCoreClockRate(const unsigned int& core) const {
			std::ifstream inputStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/cpuinfo_max_freq");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Hertz(std::stoi(cpuInfo), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Watt CPU::getPowerConsumption() const {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			return powerConsumption_;
		}

		unsigned int CPU::getCoreCount() const {
			return getProcCPUInfoValuesPerCPU()[id_].size();
		}

		std::chrono::system_clock::duration CPU::getUserTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getUserTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getUserTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "userTimespan");
		}

		std::chrono::system_clock::duration CPU::getNiceTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getNiceTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getNiceTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "niceTimespan");
		}

		std::chrono::system_clock::duration CPU::getSystemTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getSystemTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getSystemTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "systemTimespan");
		}

		std::chrono::system_clock::duration CPU::getIdleTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getIdleTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getIdleTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "idleTimespan");
		}

		std::chrono::system_clock::duration CPU::getIOWaitTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getIOWaitTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getIOWaitTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "ioWaitTimespan");
		}

		std::chrono::system_clock::duration CPU::getInterruptsTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getInterruptsTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getInterruptsTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "interruptsTimespan");
		}

		std::chrono::system_clock::duration CPU::getSoftInterruptsTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getSoftInterruptsTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getSoftInterruptsTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "softInterruptsTimespan");
		}

		std::chrono::system_clock::duration CPU::getStealTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getStealTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getStealTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "stealTimespan");
		}

		std::chrono::system_clock::duration CPU::getGuestTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getGuestTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getGuestTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "guestTimespan");
		}

		std::chrono::system_clock::duration CPU::getGuestNiceTimespan() const {
			std::chrono::system_clock::duration sum;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getGuestNiceTimespan(coreIndex);
			}

			return sum;
		}

		std::chrono::system_clock::duration CPU::getGuestNiceTimespan(const unsigned int& core) const {
			return getProcStatTimespan(core, "guestNiceTimespan");
		}
	}
}
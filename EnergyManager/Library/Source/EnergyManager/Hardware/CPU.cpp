#include "./CPU.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
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
		std::mutex CPU::monitorThreadMutex_;

		std::map<unsigned int, std::map<std::string, std::string>> CPU::getProcCPUInfoValuesPerProcessor() {
			// Keep track of each access time
			static std::chrono::system_clock::time_point lastProcCPUInfoValuesPerProcessorRetrieval = std::chrono::system_clock::now();

			// Keep track of the last values
			static std::map<unsigned int, std::map<std::string, std::string>> procCPUInfoValues = {};

			// Set up a mutex
			static std::mutex procCPUInfoValuesMutex;
			std::lock_guard<std::mutex> guard(procCPUInfoValuesMutex);

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
						procCPUInfoValues[currentProcessorID][valuePair.front()] = Utility::Text::trim(valuePair.back());
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

				//auto coreID = std::stoi(processorValues.second["core id"]);
				auto coreID = processorValues.first;
				cpuCoreValues[cpuID][coreID] = processorValues.second;
			}

			return cpuCoreValues;
		}

		std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> CPU::getProcStatValuesPerProcessor() {
			// Keep track of each access time
			static std::chrono::system_clock::time_point lastProcStatValuesRetrieval = std::chrono::system_clock::now();

			// Keep track of the last values
			static std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> procStatValues = {};

			// Set up a mutex
			static std::mutex procStatValuesMutex;
			std::lock_guard<std::mutex> guard(procStatValuesMutex);

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

						procStatValues[processorID][name] = std::chrono::duration_cast<std::chrono::system_clock::duration>(
							std::chrono::milliseconds(static_cast<unsigned long>(std::stol(processorInfoValues[valueIndex]) / jiffiesPerMillisecond)));
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

				//auto coreID = std::stoi(procCPUInfoValuesPerProcessor[processorID]["core id"]);
				auto coreID = processorID;
				cpuCoreValues[cpuID][coreID] = processorValues.second;
			}

			return cpuCoreValues;
		}

		CPU::CPU(const unsigned int& id) : id_(id) {
			monitorThread_ = std::thread([&] {
				// Record the last time at which the variables were polled
				std::chrono::system_clock::time_point lastMonitorTimestamp = std::chrono::system_clock::now();

				// Record the last `/proc/stat` values
				std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> lastProcStatValues = getProcStatValuesPerCPU();

				// Record the last polled energy consumption
				Utility::Units::Joule lastEnergyConsumption = 0;

				// Keep the thread running
				while(monitorThreadRunning_) {
					auto currentTimestamp = std::chrono::system_clock::now();

					if((currentTimestamp - lastMonitorTimestamp) >= std::chrono::milliseconds(100)) {
						std::lock_guard<std::mutex> guard(monitorThreadMutex_);

						// Get the current values
						auto currentProcStatValues = getProcStatValuesPerCPU();
						auto currentEnergyConsumption = getEnergyConsumption();
						auto pollingTimespan = std::chrono::duration_cast<std::chrono::milliseconds>(currentTimestamp - lastMonitorTimestamp).count() / static_cast<float>(1000);

						// Calculate the power consumption in Watts
						auto divisor = (pollingTimespan == 0 ? 0.1 : pollingTimespan);
						powerConsumption_ = (currentEnergyConsumption - lastEnergyConsumption).toValue() / divisor;

						// Calculate the core utilization rates
						for(unsigned int core = 0; core < getCoreCount(); ++core) {
							auto previousIdle = lastProcStatValues[id_][core]["idleTimespan"] + lastProcStatValues[id_][core]["ioWaitTimespan"];
							auto previousActive = lastProcStatValues[id_][core]["userTimespan"] + lastProcStatValues[id_][core]["niceTimespan"] + lastProcStatValues[id_][core]["systemTimespan"]
												  + lastProcStatValues[id_][core]["interruptsTimespan"] + lastProcStatValues[id_][core]["softInterruptsTimespan"]
												  + lastProcStatValues[id_][core]["stealTimespan"] + lastProcStatValues[id_][core]["guestTimespan"]
												  + lastProcStatValues[id_][core]["guestNiceTimespan"];
							auto previousTotal = previousIdle + previousActive;

							auto idle = currentProcStatValues[id_][core]["idleTimespan"] + currentProcStatValues[id_][core]["ioWaitTimespan"];
							auto active = currentProcStatValues[id_][core]["userTimespan"] + currentProcStatValues[id_][core]["niceTimespan"] + currentProcStatValues[id_][core]["systemTimespan"]
										  + currentProcStatValues[id_][core]["interruptsTimespan"] + currentProcStatValues[id_][core]["softInterruptsTimespan"]
										  + currentProcStatValues[id_][core]["stealTimespan"] + currentProcStatValues[id_][core]["guestTimespan"]
										  + currentProcStatValues[id_][core]["guestNiceTimespan"];
							auto total = idle + active;

							auto totalDifference = total - previousTotal;
							auto idleDifference = idle - previousIdle;
							auto activeDifference = active - previousActive;

							coreUtilizationRates_[core] = totalDifference.count() == 0 ? 0 : (static_cast<double>(activeDifference.count()) / static_cast<double>(totalDifference.count()) * 100);
						}

						// Set the variables for the next poll cycle
						lastProcStatValues = currentProcStatValues;
						lastEnergyConsumption = currentEnergyConsumption;
						lastMonitorTimestamp = currentTimestamp;
					} else {
						usleep(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::milliseconds(10)).count());
					}
				}
			});

			startEnergyConsumption_ = getEnergyConsumption();
		}

		unsigned int CPU::getProcessorID(const unsigned int& core) const {
			for(auto& procCpuInfoValuesPerProcessor : getProcCPUInfoValuesPerProcessor()) {
				if(std::stoi(procCpuInfoValuesPerProcessor.second["physical id"]) == id_ && procCpuInfoValuesPerProcessor.first == core) {
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

		std::vector<std::shared_ptr<Hardware::CPU>> CPU::parseCPUs(const std::string& cpuString) {
			std::vector<std::string> cpuStrings = EnergyManager::Utility::Text::splitToVector(cpuString, ",", true);
			std::vector<std::shared_ptr<EnergyManager::Hardware::CPU>> cpus;
			std::transform(cpuStrings.begin(), cpuStrings.end(), std::back_inserter(cpus), [](const auto& cpuString) {
				return EnergyManager::Hardware::CPU::getCPU(std::stoi(cpuString));
			});

			return cpus;
		}

		std::shared_ptr<CPU> CPU::getCPU(const unsigned int& id) {
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

		CPU::~CPU() {
			// Stop the monitor
			monitorThreadRunning_ = false;
			monitorThread_.join();
		}

		unsigned int CPU::getID() const {
			return id_;
		}

		Utility::Units::Hertz CPU::getCoreClockRate() const {
			Utility::Units::Hertz sum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				sum += getCoreClockRate(coreIndex);
			}

			return sum / getCoreCount();
		}

		Utility::Units::Hertz CPU::getCurrentMinimumCoreClockRate() const {
			Utility::Units::Hertz minimum = -1;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				auto current = getCurrentMinimumCoreClockRate(coreIndex);
				if(minimum < 0 || current < minimum) {
					minimum = current;
				}
			}

			return minimum;
		}

		Utility::Units::Hertz CPU::getCurrentMaximumCoreClockRate() const {
			Utility::Units::Hertz maximum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				auto current = getCurrentMinimumCoreClockRate(coreIndex);
				if(current > maximum) {
					maximum = current;
				}
			}

			return maximum;
		}

		Utility::Units::Hertz CPU::getCoreClockRate(const unsigned int& core) const {
			std::ifstream coreClockRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/scaling_cur_freq");
			std::string coreClockRateString((std::istreambuf_iterator<char>(coreClockRateStream)), std::istreambuf_iterator<char>());

			// FIXME: Something may not be right with this output (see visualization results)
			return Utility::Units::Hertz(std::stoul(coreClockRateString), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Hertz CPU::getCurrentMinimumCoreClockRate(const unsigned int& core) const {
			if(getPowerScalingDriver(0) == "intel_pstate") {
				std::ifstream minimumCoreClockRateStream("/sys/devices/system/cpu/intel_pstate/min_perf_pct");
				std::string minimumCoreClockRateString((std::istreambuf_iterator<char>(minimumCoreClockRateStream)), std::istreambuf_iterator<char>());

				return Utility::Units::Hertz(std::stod(minimumCoreClockRateString) / 100 * getMaximumCoreClockRate().toValue(), Utility::Units::SIPrefix::KILO);
			} else {
				std::ifstream minimumCoreClockRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/scaling_min_freq");
				std::string minimumCoreClockRateString((std::istreambuf_iterator<char>(minimumCoreClockRateStream)), std::istreambuf_iterator<char>());

				return Utility::Units::Hertz(std::stoul(minimumCoreClockRateString), Utility::Units::SIPrefix::KILO);
			}
		}

		Utility::Units::Hertz CPU::getCurrentMaximumCoreClockRate(const unsigned int& core) const {
			if(getPowerScalingDriver(0) == "intel_pstate") {
				std::ifstream maximumCoreClockRateStream("/sys/devices/system/cpu/intel_pstate/max_perf_pct");
				std::string maximumCoreClockRateString((std::istreambuf_iterator<char>(maximumCoreClockRateStream)), std::istreambuf_iterator<char>());

				return Utility::Units::Hertz(std::stod(maximumCoreClockRateString) / 100 * getMaximumCoreClockRate().toValue(), Utility::Units::SIPrefix::KILO);
			} else {
				std::ifstream maximumCoreClockRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/scaling_max_freq");
				std::string maximumCoreClockRateString((std::istreambuf_iterator<char>(maximumCoreClockRateStream)), std::istreambuf_iterator<char>());

				return Utility::Units::Hertz(std::stoul(maximumCoreClockRateString), Utility::Units::SIPrefix::KILO);
			}
		}

		void CPU::setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) {
			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				setCoreClockRate(coreIndex, minimumRate, maximumRate);
			}

			// Check the type of power scaling driver in use
			if(getPowerScalingDriver(0) == "intel_pstate") {
				// Set minimum rate
				std::ofstream minimumRateStream("/sys/devices/system/cpu/intel_pstate/min_perf_pct");
				minimumRateStream << static_cast<unsigned int>((minimumRate.toValue() / getMaximumCoreClockRate().toValue()) * 100);

				// Set maximum rate
				std::ofstream maximumRateStream("/sys/devices/system/cpu/intel_pstate/max_perf_pct");
				maximumRateStream << static_cast<unsigned int>((maximumRate.toValue() / getMaximumCoreClockRate().toValue()) * 100);
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

			// Check the type of power scaling driver in use
			if(getPowerScalingDriver(0) == "intel_pstate") {
				// Set minimum rate
				std::ofstream minimumRateStream("/sys/devices/system/cpu/intel_pstate/min_perf_pct");
				minimumRateStream << 0;

				// Set maximum rate
				std::ofstream maximumRateStream("/sys/devices/system/cpu/intel_pstate/max_perf_pct");
				maximumRateStream << 100;
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

		unsigned int CPU::getCoreCount() const {
			return getProcCPUInfoValuesPerCPU()[id_].size();
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

		Utility::Units::Hertz CPU::getMinimumCoreClockRate() const {
			Utility::Units::Hertz minimum = 0;
			bool found = false;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				auto currentCoreClockRate = getMinimumCoreClockRate(coreIndex);
				if(!found || currentCoreClockRate < minimum) {
					minimum = currentCoreClockRate;
					found = true;
				}
			}

			return minimum;
		}

		Utility::Units::Hertz CPU::getMaximumCoreClockRate() const {
			Utility::Units::Hertz maximum = 0;

			for(unsigned int coreIndex = 0u; coreIndex < getCoreCount(); ++coreIndex) {
				auto currentCoreClockRate = getMaximumCoreClockRate(coreIndex);
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

		Utility::Units::Hertz CPU::getMinimumCoreClockRate(const unsigned int& core) const {
			std::ifstream inputStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/cpuinfo_min_freq");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Hertz(std::stoi(cpuInfo), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Watt CPU::getPowerConsumption() const {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			return powerConsumption_;
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

		std::string CPU::getPowerScalingDriver(const unsigned int& core) const {
			std::ifstream powerScalingDriverStream("/sys/devices/system/cpu/cpu" + std::to_string(getProcessorID(core)) + "/cpufreq/scaling_driver");
			std::string powerScalingDriver((std::istreambuf_iterator<char>(powerScalingDriverStream)), std::istreambuf_iterator<char>());

			return Utility::Text::trim(powerScalingDriver);
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
			std::ofstream outputStream("/sys/devices/system/cpu/intel_pstate/no_turbo");
			outputStream << !turbo;
		}
	}
}
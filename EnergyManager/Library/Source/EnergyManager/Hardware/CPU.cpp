#include "./CPU.hpp"

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
				auto processorID = processorValues.first;
				cpuCoreValues[cpuID][processorID] = processorValues.second;
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
							std::chrono::milliseconds(static_cast<unsigned long>(std::stoul(processorInfoValues[valueIndex]) / jiffiesPerMillisecond)));
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

		CPU::CPU(const unsigned int& id) : CentralProcessor(id), Utility::Loopable(std::chrono::milliseconds(100)) {
			// Detect and add all cores
			const auto data = getProcCPUInfoValuesPerCPU()[getID()];
			auto coreID = 0;
			for(const auto& coreDataElement : data) {
				cores_.push_back(std::shared_ptr<Core>(new Core(this, coreDataElement.first, coreID++)));
			}

			startEnergyConsumption_ = getEnergyConsumption();

			// start the monitor thread
			run(true);
		}

		std::vector<std::string> CPU::generateHeaders() const {
			auto headers = Loopable::generateHeaders();
			headers.push_back("CPU " + Utility::Text::toString(getID()));

			return headers;
		}

		void CPU::onLoop() {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			// Get the timestamp
			auto currentTimestamp = std::chrono::system_clock::now();

			// Get the time since last poll in seconds, with decimals
			auto pollingTimespan = std::chrono::duration<double>(currentTimestamp - lastMonitorTimestamp_).count();

			// Get the current values
			auto currentProcStatValues = getProcStatValuesPerCPU();
			auto currentEnergyConsumption = getEnergyConsumption();

			// Calculate the power consumption in Watts
			if(pollingTimespan == 0) {
				logWarning("Polling timespan equal to zero, can't measure power consumption");
			} else {
				auto energyConsumed = currentEnergyConsumption - lastEnergyConsumption_;
				auto currentPowerConsumption = Utility::Units::Watt(energyConsumed.toValue() / pollingTimespan);

				// FIXME: There can be some noise in the data so we filter out values that are too high
				if(currentEnergyConsumption.toValue() < 300) {
					powerConsumption_ = currentPowerConsumption;
				}
			}

			// Calculate the core utilization rates
			for(unsigned int core = 0; core < getCores().size(); ++core) {
				auto previousIdle = lastProcStatValues_[getID()][core]["idleTimespan"] + lastProcStatValues_[getID()][core]["ioWaitTimespan"];
				auto previousActive = lastProcStatValues_[getID()][core]["userTimespan"] + lastProcStatValues_[getID()][core]["niceTimespan"] + lastProcStatValues_[getID()][core]["systemTimespan"]
									  + lastProcStatValues_[getID()][core]["interruptsTimespan"] + lastProcStatValues_[getID()][core]["softInterruptsTimespan"]
									  + lastProcStatValues_[getID()][core]["stealTimespan"] + lastProcStatValues_[getID()][core]["guestTimespan"]
									  + lastProcStatValues_[getID()][core]["guestNiceTimespan"];
				auto previousTotal = previousIdle + previousActive;

				auto idle = currentProcStatValues[getID()][core]["idleTimespan"] + currentProcStatValues[getID()][core]["ioWaitTimespan"];
				auto active = currentProcStatValues[getID()][core]["userTimespan"] + currentProcStatValues[getID()][core]["niceTimespan"] + currentProcStatValues[getID()][core]["systemTimespan"]
							  + currentProcStatValues[getID()][core]["interruptsTimespan"] + currentProcStatValues[getID()][core]["softInterruptsTimespan"]
							  + currentProcStatValues[getID()][core]["stealTimespan"] + currentProcStatValues[getID()][core]["guestTimespan"]
							  + currentProcStatValues[getID()][core]["guestNiceTimespan"];
				auto total = idle + active;

				auto totalDifference = total - previousTotal;
				auto idleDifference = idle - previousIdle;
				auto activeDifference = active - previousActive;

				coreUtilizationRates_[core] = totalDifference.count() == 0 ? 0 : (static_cast<double>(activeDifference.count()) / static_cast<double>(totalDifference.count()) * 100);
			}

			// Set the variables for the next poll cycle
			lastMonitorTimestamp_ = currentTimestamp;
			lastProcStatValues_ = currentProcStatValues;
			lastEnergyConsumption_ = currentEnergyConsumption;
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

		std::vector<std::shared_ptr<CPU::Core>> CPU::getCores() const {
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

		Utility::Units::Percent CPU::getCoreUtilizationRate() const {
			double sum = 0;

			for(const auto& core : getCores()) {
				sum += core->getCoreUtilizationRate().getUnit();
			}

			return sum / getCores().size();
		}

		Utility::Units::Joule CPU::getEnergyConsumption() const {
			std::ifstream inputStream("/sys/class/powercap/intel-rapl/intel-rapl:" + std::to_string(getID()) + "/energy_uj");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Joule(std::stod(cpuInfo), Utility::Units::SIPrefix::MICRO) - startEnergyConsumption_;
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

		Utility::Units::Watt CPU::getPowerConsumption() const {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			return powerConsumption_;
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

		std::chrono::system_clock::duration CPU::Core::getProcStatTimespan(const std::string& name) const {
			std::lock_guard<std::mutex> guard(monitorThreadMutex_);

			const auto lastValue = getProcStatValuesPerCPU().at(getCPU()->getID()).at(getID()).at(name);
			const auto startValue = getCPU()->startProcStatValues_.at(getCPU()->getID()).at(getID()).at(name);

			// Clamp the value to be larger than 0
			const auto difference = lastValue - startValue;
			return difference >= std::chrono::system_clock::duration(0) ? difference : std::chrono::system_clock::duration(0);
		}

		CPU::Core::Core(CPU* cpu, const unsigned int& id, const unsigned int& coreID) : CentralProcessor(id), cpu_(cpu), coreID_(coreID) {
		}

		std::vector<std::string> CPU::Core::generateHeaders() const {
			return { "CPU " + Utility::Text::toString(cpu_->getID()), "Core " + Utility::Text::toString(getID()) };
		}

		std::shared_ptr<CPU::Core> CPU::Core::getCore(const unsigned int& id) {
			for(const auto& cpu : CPU::getCPUs()) {
				for(const auto& core : cpu->getCores()) {
					if(core->getID() == id) {
						return core;
					}
				}
			}

			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not find core");
		}

		std::shared_ptr<CPU> CPU::Core::getCPU() const {
			return CPU::getCPU(cpu_->getID());
		}

		unsigned int CPU::Core::getCoreID() const {
			return coreID_;
		}

		Utility::Units::Hertz CPU::Core::getCoreClockRate() const {
			std::ifstream coreClockRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/scaling_cur_freq");
			std::string coreClockRateString((std::istreambuf_iterator<char>(coreClockRateStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Hertz(std::stoul(coreClockRateString), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Hertz CPU::Core::getCurrentMinimumCoreClockRate() const {
			if(getPowerScalingDriver() == "intel_pstate") {
				std::ifstream minimumCoreClockRateStream("/sys/devices/system/cpu/intel_pstate/min_perf_pct");
				std::string minimumCoreClockRateString((std::istreambuf_iterator<char>(minimumCoreClockRateStream)), std::istreambuf_iterator<char>());
				const auto minimumCoreClockRatePercentage = std::stod(minimumCoreClockRateString);

				return Utility::Units::Hertz(minimumCoreClockRatePercentage / 100 * getMaximumCoreClockRate().toValue());
			} else {
				std::ifstream minimumCoreClockRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/scaling_min_freq");
				std::string minimumCoreClockRateString((std::istreambuf_iterator<char>(minimumCoreClockRateStream)), std::istreambuf_iterator<char>());

				return Utility::Units::Hertz(std::stoul(minimumCoreClockRateString), Utility::Units::SIPrefix::KILO);
			}
		}

		Utility::Units::Hertz CPU::Core::getCurrentMaximumCoreClockRate() const {
			if(getPowerScalingDriver() == "intel_pstate") {
				std::ifstream maximumCoreClockRateStream("/sys/devices/system/cpu/intel_pstate/max_perf_pct");
				std::string maximumCoreClockRateString((std::istreambuf_iterator<char>(maximumCoreClockRateStream)), std::istreambuf_iterator<char>());
				const auto maximumCoreClockRatePercentage = std::stod(maximumCoreClockRateString);
				const auto maximumCoreClockRate = static_cast<unsigned long>(maximumCoreClockRatePercentage / 100.0) * getMaximumCoreClockRate();

				return Utility::Units::Hertz(maximumCoreClockRatePercentage / 100 * getMaximumCoreClockRate().toValue());
			} else {
				std::ifstream maximumCoreClockRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/scaling_max_freq");
				std::string maximumCoreClockRateString((std::istreambuf_iterator<char>(maximumCoreClockRateStream)), std::istreambuf_iterator<char>());

				return Utility::Units::Hertz(std::stoul(maximumCoreClockRateString), Utility::Units::SIPrefix::KILO);
			}
		}

		void CPU::Core::setCoreClockRate(const Utility::Units::Hertz& minimumRate, const Utility::Units::Hertz& maximumRate) {
			logDebug("Setting clock rate range to [%lu, %lu]...", minimumRate.toValue(), minimumRate.toValue());

			// Set minimum rate
			std::ofstream minimumRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/scaling_min_freq");
			minimumRateStream << minimumRate.convertPrefix(Utility::Units::SIPrefix::KILO);

			// Set maximum rate
			std::ofstream maximumRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/scaling_max_freq");
			maximumRateStream << maximumRate.convertPrefix(Utility::Units::SIPrefix::KILO);
		}

		void CPU::Core::resetCoreClockRate() {
			logDebug("Resetting clock rate...");

			std::ifstream minimumRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/cpuinfo_min_freq");
			std::string minimumRateString((std::istreambuf_iterator<char>(minimumRateStream)), std::istreambuf_iterator<char>());
			Utility::Units::Hertz minimumRate(std::stoul(minimumRateString), Utility::Units::SIPrefix::KILO);

			std::ifstream maximumRateStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/cpuinfo_max_freq");
			std::string maximumRateString((std::istreambuf_iterator<char>(maximumRateStream)), std::istreambuf_iterator<char>());
			Utility::Units::Hertz maximumRate(std::stoul(maximumRateString), Utility::Units::SIPrefix::KILO);

			setCoreClockRate(minimumRate, maximumRate);
		}

		Utility::Units::Percent CPU::Core::getCoreUtilizationRate() const {
			std::lock_guard<std::mutex> guard(getCPU()->monitorThreadMutex_);

			return getCPU()->coreUtilizationRates_.at(getCoreID());
		}

		Utility::Units::Hertz CPU::Core::getMaximumCoreClockRate() const {
			std::ifstream inputStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/cpuinfo_max_freq");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Hertz(std::stoi(cpuInfo), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Hertz CPU::Core::getMinimumCoreClockRate() const {
			std::ifstream inputStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/cpuinfo_min_freq");
			std::string cpuInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

			return Utility::Units::Hertz(std::stoi(cpuInfo), Utility::Units::SIPrefix::KILO);
		}

		Utility::Units::Joule CPU::Core::getEnergyConsumption() const {
			// TODO: Maybe there is a way to measure the energy consumption of individual cores
			return getCPU()->getEnergyConsumption() / getCPU()->getCores().size();
		}

		Utility::Units::Watt CPU::Core::getPowerConsumption() const {
			// TODO: Maybe there is a way to measure the power consumption of individual cores
			return getCPU()->getPowerConsumption() / getCPU()->getCores().size();
		}

		Utility::Units::Celsius CPU::Core::getTemperature() const {
			// TODO: Maybe there is a way to measure the temperature of individual cores
			return getCPU()->getTemperature();
		}

		std::chrono::system_clock::duration CPU::Core::getUserTimespan() const {
			return getProcStatTimespan("userTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getNiceTimespan() const {
			return getProcStatTimespan("niceTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getSystemTimespan() const {
			return getProcStatTimespan("systemTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getIdleTimespan() const {
			return getProcStatTimespan("idleTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getIOWaitTimespan() const {
			return getProcStatTimespan("ioWaitTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getInterruptsTimespan() const {
			return getProcStatTimespan("interruptsTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getSoftInterruptsTimespan() const {
			return getProcStatTimespan("softInterruptsTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getStealTimespan() const {
			return getProcStatTimespan("stealTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getGuestTimespan() const {
			return getProcStatTimespan("guestTimespan");
		}

		std::chrono::system_clock::duration CPU::Core::getGuestNiceTimespan() const {
			return getProcStatTimespan("guestNiceTimespan");
		}

		std::string CPU::Core::getPowerScalingDriver() const {
			std::ifstream powerScalingDriverStream("/sys/devices/system/cpu/cpu" + std::to_string(getID()) + "/cpufreq/scaling_driver");
			std::string powerScalingDriver((std::istreambuf_iterator<char>(powerScalingDriverStream)), std::istreambuf_iterator<char>());

			return Utility::Text::trim(powerScalingDriver);
		}
	}
}
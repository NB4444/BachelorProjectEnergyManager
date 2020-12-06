#include "./CentralProcessor.hpp"

#include "EnergyManager/Utility/CachedValue.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Hardware {
		std::map<unsigned int, std::map<std::string, std::string>> CentralProcessor::getProcCPUInfoValuesPerProcessor() {
			static Utility::CachedValue<std::map<unsigned int, std::map<std::string, std::string>>> procCPUInfoValuesPerProcessor(std::chrono::milliseconds(100));

			return procCPUInfoValuesPerProcessor.getValue([](const auto& value, const auto& timeSinceLastUpdate) {
				std::map<unsigned int, std::map<std::string, std::string>> result;

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
						result[currentProcessorID][valuePair.front()] = Utility::Text::trim(valuePair.back());
					}
				}

				return result;
			});
		}

		std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> CentralProcessor::getProcCPUInfoValuesPerCPU() {
			// Keep track of the values
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> cpuCoreValues;

			// Parse the values per CPU
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

		std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> CentralProcessor::getProcStatValuesPerProcessor() {
			static Utility::CachedValue<std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> procStatValuesPerProcessor(std::chrono::milliseconds(100));

			return procStatValuesPerProcessor.getValue([](const auto& value, const auto& timeSinceLastUpdate) {
				std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> result;

				// Read the CPU info
				auto processorInfo = Utility::Text::readFile("/proc/stat");

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

						result[processorID][name] = std::chrono::duration_cast<std::chrono::system_clock::duration>(
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

				return result;
			});
		}

		std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> CentralProcessor::getProcStatValuesPerCPU() {
			// Keep track of the last values
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> cpuCoreValues;

			auto procCPUInfoValuesPerProcessor = getProcCPUInfoValuesPerProcessor();
			auto procStatValuesPerProcessor = getProcStatValuesPerProcessor();

			// Parse the values per CPU
			for(auto& processorValues : procStatValuesPerProcessor) {
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
	}
}
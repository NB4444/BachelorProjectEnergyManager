#include "./CentralProcessor.hpp"

#include "EnergyManager/Utility/Text.hpp"

#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace EnergyManager {
	namespace Hardware {
		std::map<unsigned int, std::map<std::string, std::string>> CentralProcessor::getProcCentralProcessorInfoValuesPerProcessor() {
			// Keep track of each access time
			static std::chrono::system_clock::time_point lastProcCentralProcessorInfoValuesPerProcessorRetrieval = std::chrono::system_clock::now();

			// Keep track of the last values
			static std::map<unsigned int, std::map<std::string, std::string>> procCentralProcessorInfoValues = {};

			// Set up a mutex
			static std::mutex procCentralProcessorInfoValuesMutex;
			std::lock_guard<std::mutex> guard(procCentralProcessorInfoValuesMutex);

			if(procCentralProcessorInfoValues.empty()
			   || std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastProcCentralProcessorInfoValuesPerProcessorRetrieval).count() > 100) {
				procCentralProcessorInfoValues.clear();
				lastProcCentralProcessorInfoValuesPerProcessorRetrieval = std::chrono::system_clock::now();

				// Read the CentralProcessor info
				std::ifstream inputStream("/proc/CentralProcessorinfo");
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
						// Set the current CentralProcessor ID
						currentProcessorID = std::stoi(valuePair.back());
					} else {
						// Set the variable
						procCentralProcessorInfoValues[currentProcessorID][valuePair.front()] = Utility::Text::trim(valuePair.back());
					}
				}
			}

			return procCentralProcessorInfoValues;
		}

		std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> CentralProcessor::getProcCentralProcessorInfoValuesPerCentralProcessor() {
			// Parse the values per CentralProcessor
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::string>>> CentralProcessorCoreValues = {};
			for(auto& processorValues : getProcCentralProcessorInfoValuesPerProcessor()) {
				auto CentralProcessorID = std::stoi(processorValues.second["physical id"]);

				// Create structures if they don't exist
				if(CentralProcessorCoreValues.find(CentralProcessorID) == CentralProcessorCoreValues.end()) {
					CentralProcessorCoreValues[CentralProcessorID] = {};
				}

				//auto coreID = std::stoi(processorValues.second["core id"]);
				auto coreID = processorValues.first;
				CentralProcessorCoreValues[CentralProcessorID][coreID] = processorValues.second;
			}

			return CentralProcessorCoreValues;
		}

		std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>> CentralProcessor::getProcStatValuesPerProcessor() {
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

				// Read the CentralProcessor info
				std::ifstream inputStream("/proc/stat");
				std::string processorInfo((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());

				// Parse the values to lines
				std::vector<std::string> processorInfoLines = Utility::Text::splitToVector(processorInfo, "\n");

				// Parse the lines
				for(const auto& processorInfoLine : processorInfoLines) {
					// Only process processors
					if(processorInfoLine.rfind("CentralProcessor", 0) != 0) {
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

		std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> CentralProcessor::getProcStatValuesPerCentralProcessor() {
			auto procCentralProcessorInfoValuesPerProcessor = getProcCentralProcessorInfoValuesPerProcessor();

			// Parse the values per CentralProcessor
			std::map<unsigned int, std::map<unsigned int, std::map<std::string, std::chrono::system_clock::duration>>> CentralProcessorCoreValues;
			for(auto& processorValues : getProcStatValuesPerProcessor()) {
				auto processorID = processorValues.first;

				auto CentralProcessorID = std::stoi(procCentralProcessorInfoValuesPerProcessor[processorID]["physical id"]);

				// Create structures if they don't exist
				if(CentralProcessorCoreValues.find(CentralProcessorID) == CentralProcessorCoreValues.end()) {
					CentralProcessorCoreValues[CentralProcessorID] = {};
				}

				//auto coreID = std::stoi(procCentralProcessorInfoValuesPerProcessor[processorID]["core id"]);
				auto coreID = processorID;
				CentralProcessorCoreValues[CentralProcessorID][coreID] = processorValues.second;
			}

			return CentralProcessorCoreValues;
		}
	}
}
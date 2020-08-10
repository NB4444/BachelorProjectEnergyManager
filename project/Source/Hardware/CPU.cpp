#include "./CPU.hpp"

#include "Utility/Text.hpp"

#include <fstream>
#include <string>
#include <vector>

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
		: id_(id) {
	}

	std::shared_ptr<CPU> CPU::getCPU(const unsigned int& id) {
		auto iterator = cpus_.find(id);
		if(iterator == cpus_.end()) {
			cpus_[id] = std::shared_ptr<CPU>(new CPU(id));
		}

		return cpus_[id];
	}
}
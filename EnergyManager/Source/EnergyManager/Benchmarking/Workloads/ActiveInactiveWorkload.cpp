#include "./ActiveInactiveWorkload.hpp"

#include "EnergyManager/Benchmarking/Operations/AddGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/AllocateCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/AllocateGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/AssignCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/CopyCPUToGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/CopyGPUToCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/FreeCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/FreeGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/SleepOperation.hpp"
#include "EnergyManager/Utility/Exceptions/ParseException.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			void ActiveInactiveWorkload::initialize() {
				SyntheticWorkload::addParser([](const std::string& name, const std::map<std::string, std::string>& parameters) {
					if(name != "ActiveInactiveWorkload") {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_PARSE_EXCEPTION();
					}

					return std::make_shared<EnergyManager::Benchmarking::Workloads::ActiveInactiveWorkload>(
						std::stoi(Utility::Text::getParameter(parameters, "activeOperations")),
						std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(std::stol(Utility::Text::getParameter(parameters, "inactivePeriod")))),
						std::stoi(Utility::Text::getParameter(parameters, "cycles")));
				});
			}

			ActiveInactiveWorkload::ActiveInactiveWorkload(const unsigned int& activeOperations, const std::chrono::system_clock::duration& inactivePeriod, const unsigned int& cycles) {
				for(unsigned int cycle = 0; cycle < cycles; ++cycle) {
					for(unsigned int operation = 0; operation < activeOperations; ++operation) {
						size_t size = 1024;

						// Allocate host input vectors
						addOperation(std::make_shared<Operations::AllocateCPUOperation>(size));
						addOperation(std::make_shared<Operations::AllocateCPUOperation>(size));
						addOperation(std::make_shared<Operations::AssignCPUOperation>(2));

						// Allocate device input vectors
						addOperation(std::make_shared<Operations::AllocateGPUOperation>(size));
						addOperation(std::make_shared<Operations::AllocateGPUOperation>(size));
						addOperation(std::make_shared<Operations::CopyCPUToGPUOperation>(2));
						addOperation(std::make_shared<Operations::FreeCPUOperation>());
						addOperation(std::make_shared<Operations::FreeCPUOperation>());

						// Do vector add
						addOperation(std::make_shared<Operations::AllocateGPUOperation>(size));
						addOperation(std::make_shared<Operations::AddGPUOperation>(2, 32));

						// Copy results to host
						addOperation(std::make_shared<Operations::AllocateCPUOperation>(size));
						addOperation(std::make_shared<Operations::CopyGPUToCPUOperation>(1));
						addOperation(std::make_shared<Operations::FreeGPUOperation>());
						addOperation(std::make_shared<Operations::FreeCPUOperation>());

						// Free device vectors
						addOperation(std::make_shared<Operations::FreeGPUOperation>());
						addOperation(std::make_shared<Operations::FreeGPUOperation>());
					}

					// Sleep
					addOperation(std::make_shared<Operations::SleepOperation>(inactivePeriod));
				}
			}
		}
	}
}
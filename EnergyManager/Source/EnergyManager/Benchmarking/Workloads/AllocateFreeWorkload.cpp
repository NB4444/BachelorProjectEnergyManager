#include "./AllocateFreeWorkload.hpp"

#include "EnergyManager/Benchmarking/Operations/AllocateCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/AllocateGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/FreeCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/FreeGPUOperation.hpp"
#include "EnergyManager/Utility/Exceptions/ParseException.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			void AllocateFreeWorkload::initialize() {
				SyntheticWorkload::addParser([](const std::string& name, const std::map<std::string, std::string>& parameters) {
					if(name != "ActiveInactiveWorkload") {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_PARSE_EXCEPTION();
					}

					return std::make_shared<EnergyManager::Benchmarking::Workloads::AllocateFreeWorkload>(
						std::stoi(Utility::Text::getParameter(parameters, "hostAllocations")),
						std::stol(Utility::Text::getParameter(parameters, "hostSize")),
						std::stoi(Utility::Text::getParameter(parameters, "deviceAllocations")),
						std::stol(Utility::Text::getParameter(parameters, "deviceSize")));
				});
			}

			AllocateFreeWorkload::AllocateFreeWorkload(const unsigned int& hostAllocations, const size_t& hostSize, const unsigned int& deviceAllocations, const size_t& deviceSize) {
				// Allocate host vectors
				for(unsigned int index = 0; index < hostAllocations; ++index) {
					addOperation(std::make_shared<Operations::AllocateCPUOperation>(hostSize));
				}

				// Allocate device vectors
				for(unsigned int index = 0; index < deviceAllocations; ++index) {
					addOperation(std::make_shared<Operations::AllocateGPUOperation>(deviceSize));
				}

				// Free device vectors
				for(unsigned int index = 0; index < deviceAllocations; ++index) {
					addOperation(std::make_shared<Operations::FreeGPUOperation>());
				}

				// Free host vectors
				for(unsigned int index = 0; index < hostAllocations; ++index) {
					addOperation(std::make_shared<Operations::FreeCPUOperation>());
				}
			}
		}
	}
}
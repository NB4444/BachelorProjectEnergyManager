#include "./VectorAddWorkload.hpp"

#include "EnergyManager/Benchmarking/Operations/AddGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/AllocateCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/AllocateGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/AssignCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/CopyCPUToGPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/CopyGPUToCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/FreeCPUOperation.hpp"
#include "EnergyManager/Benchmarking/Operations/FreeGPUOperation.hpp"
#include "EnergyManager/Utility/Exceptions/ParseException.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			void VectorAddWorkload::initialize() {
				SyntheticWorkload::addParser([](const std::string& name, const std::map<std::string, std::string>& parameters) {
					if(name != "ActiveInactiveWorkload") {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_PARSE_EXCEPTION();
					}

					return std::make_shared<EnergyManager::Benchmarking::Workloads::VectorAddWorkload>(std::stol(Utility::Text::getParameter(parameters, "size")));
				});
			}

			VectorAddWorkload::VectorAddWorkload(const size_t& size) {
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
		}
	}
}
#pragma once

#include "EnergyManager/Benchmarking/Operations/SyntheticOperation.hpp"
#include "EnergyManager/Hardware/GPU.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class SyntheticWorkload {
				std::vector<std::shared_ptr<Operations::SyntheticOperation>> operations_;

			public:
				SyntheticWorkload(std::vector<std::shared_ptr<Operations::SyntheticOperation>> operations = {});

				void addOperation(const std::shared_ptr<Operations::SyntheticOperation>& operation);

				void run(const std::shared_ptr<Hardware::GPU>& gpu);
			};
		}
	}
}
#pragma once

#include "EnergyManager/Benchmarking/Operations/SyntheticOperation.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <memory>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			class SyntheticWorkload {
				std::vector<std::shared_ptr<Operations::SyntheticOperation>> operations_;

			public:
				SyntheticWorkload(std::vector<std::shared_ptr<Operations::SyntheticOperation>> operations = {});

				void addOperation(const std::shared_ptr<Operations::SyntheticOperation>& operation);

				void run();
			};
		}
	}
}
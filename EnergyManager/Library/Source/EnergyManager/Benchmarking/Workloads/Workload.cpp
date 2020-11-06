#include "./Workload.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			void Workload::onRun() {
				for(const auto& operation : operations_) {
					operation->run();
				}
			}

			Workload::Workload(std::vector<std::shared_ptr<Operations::Operation>> operations) : operations_(std::move(operations)) {
			}

			void Workload::addOperation(const std::shared_ptr<Operations::Operation>& operation) {
				operations_.emplace_back(operation);
			}
		}
	}
}
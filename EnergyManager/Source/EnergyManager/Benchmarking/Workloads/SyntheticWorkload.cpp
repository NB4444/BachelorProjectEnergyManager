#include "./SyntheticWorkload.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			SyntheticWorkload::SyntheticWorkload(std::vector<std::shared_ptr<Operations::SyntheticOperation>> operations) : operations_(std::move(operations)) {
			}

			void SyntheticWorkload::addOperation(const std::shared_ptr<Operations::SyntheticOperation>& operation) {
				operations_.emplace_back(operation);
			}

			void SyntheticWorkload::run(const std::shared_ptr<Hardware::GPU>& gpu) {
				if(gpu != nullptr) {
					gpu->makeActive();
				}

				for(const auto& operation : operations_) {
					operation->run();
				}
			}
		}
	}
}
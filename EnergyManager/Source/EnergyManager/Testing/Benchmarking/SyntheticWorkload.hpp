#pragma once

#include <vector>
#include <string>
#include <map>

namespace EnergyManager {
	namespace Testing {
		namespace Benchmarking {
			template<typename Operation>
			class SyntheticWorkload {
				std::vector<std::pair<Operation, std::map<std::string, std::string>>> operations_;

			protected:
				virtual void processOperation(const Operation& operation, const std::map<std::string, std::string>& parameters) = 0;

			public:
				SyntheticWorkload(const std::vector<std::pair<Operation, std::map<std::string, std::string>>>& operations = {}) : operations_(operations) {
				}

				void addOperation(const Operation& operation, const std::map<std::string, std::string>& parameters) {
					operations_.emplace_back(operation, parameters);
				}

				void run() {
					for(const auto& operation : operations_) {
						processOperation(operation.first, operation.second);
					}
				}
			};
		}
	}
}
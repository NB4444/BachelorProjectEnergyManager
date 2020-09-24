#include "./SyntheticWorkload.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			std::vector<SyntheticWorkload::Parser> SyntheticWorkload::parsers_ = {};

			void SyntheticWorkload::addParser(const SyntheticWorkload::Parser& parser) {
				parsers_.push_back(parser);
			}

			std::shared_ptr<SyntheticWorkload> SyntheticWorkload::parse(const std::string& name, const std::map<std::string, std::string>& parameters) {
				for(const auto& parser : parsers_) {
					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION_IGNORE(return parser(name, parameters));
				}

				ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Could not parse SyntheticWorkload");
			}

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
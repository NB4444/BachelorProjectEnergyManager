#include "./EnergyManager.hpp"

#include "EnergyManager/Utility/Logging.hpp"

#include <thread>
#include <utility>

namespace EnergyManager {
	namespace EnergySaving {
		void EnergyManager::beforeRun() {
			// Start the strategy threads
			for(auto& strategy : strategies_) {
				Utility::Logging::logInformation("Starting strategy thread...");
				strategy->run(true);
			}
		}

		void EnergyManager::afterRun() {
			// Stop all strategy threads
			for(auto& strategy : strategies_) {
				Utility::Logging::logInformation("Stopping strategy...");
				strategy->stop(true);
			}
		}

		EnergyManager::EnergyManager(std::vector<std::shared_ptr<Strategies::Strategy>> strategies) : strategies_(std::move(strategies)) {
		}

		void EnergyManager::addStrategy(const std::shared_ptr<Strategies::Strategy>& strategy) {
			strategies_.push_back(strategy);
		}
	}
}
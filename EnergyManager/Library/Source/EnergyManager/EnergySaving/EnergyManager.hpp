#pragma once

#include "EnergyManager/EnergySaving/Strategies/Strategy.hpp"
#include "EnergyManager/Utility/Runnable.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace EnergyManager {
	namespace EnergySaving {
		/**
		 * Used to run Strategies.
		 */
		class EnergyManager : public Utility::Runnable {
			/**
			 * The Strategies to execute.
			 */
			std::vector<std::shared_ptr<Strategies::Strategy>> strategies_;

		protected:
			void beforeRun() final;

			void afterRun() final;

		public:
			/**
			 * Creates a new EnergyManager.
			 * @param strategies TheStrategies to execute.
			 */
			explicit EnergyManager(std::vector<std::shared_ptr<Strategies::Strategy>> strategies = {});

			/**
			 * Adds a Strategy to run.
			 * @param strategy The Strategy to add.
			 */
			void addStrategy(const std::shared_ptr<Strategies::Strategy>& strategy);
		};
	}
}
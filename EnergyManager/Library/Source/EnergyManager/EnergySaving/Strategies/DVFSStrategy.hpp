#pragma once

#include "EnergyManager/EnergySaving/Models/DVFSModel.hpp"
#include "EnergyManager/EnergySaving/Strategies/Strategy.hpp"
#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"

#include <memory>

namespace EnergyManager {
	namespace EnergySaving {
		namespace Strategies {
			/**
			 * A strategy that saves energy by manipulating voltages and frequencies of the available devices.
			 */
			class DVFSStrategy : public Strategy {
				/**
				 * The CPU to use.
				 */
				std::shared_ptr<Hardware::CPU> cpu_;

				/**
				 * The GPU to use.
				 */
				std::shared_ptr<Hardware::GPU> gpu_;

				/**
				 * The model that is used to predict parameters.
				 */
				const std::shared_ptr<Models::DVFSModel> model_;

			protected:
				bool isApplicable() override;

				void onUpdate() override;

			public:
				/**
				 * Creates a new DVFSStrategy.
				 * @param cpu The CPU to use.
				 * @param gpu The GPU to use.
				 * @param interval The interval at which to execute the Strategy.
				 * @param model The model to use.
				 */
				explicit DVFSStrategy(
					std::shared_ptr<Hardware::CPU> cpu,
					std::shared_ptr<Hardware::GPU> gpu,
					const std::chrono::system_clock::duration& interval,
					const std::shared_ptr<Models::DVFSModel>& model);

				/**
				 * Saves the model.
				 */
				~DVFSStrategy();
			};
		}
	}
}

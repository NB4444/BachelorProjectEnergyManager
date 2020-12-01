#pragma once

#include "EnergyManager/Utility/Loopable.hpp"

#include <chrono>

namespace EnergyManager {
	namespace EnergySaving {
		namespace Strategies {
			/**
			 * A strategy that can execute in a continuous loop to manage the current system to save energy.
			 */
			class Strategy : public Utility::Loopable {
			protected:
				void onLoop() final;

				/**
				 * Determines if the Strategy can be executed.
				 * This function is called before each update.
				 * @return Whether the Strategy can be executed.
				 */
				virtual bool isApplicable();

				/**
				 * Executes when the Strategy is updated.
				 */
				virtual void onUpdate();

			public:
				/**
				 * Creates a new Strategy.
				 * @param interval The interval at which to execute the Strategy.
				 */
				explicit Strategy(const std::chrono::system_clock::duration& interval);

				/**
				 * Performs one update of the Strategy.
				 */
				void update();
			};
		}
	}
}
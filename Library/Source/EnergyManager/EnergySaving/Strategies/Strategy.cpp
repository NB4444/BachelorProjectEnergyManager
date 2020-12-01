#include "./Strategy.hpp"

namespace EnergyManager {
	namespace EnergySaving {
		namespace Strategies {
			void Strategy::onLoop() {
				update();
			}

			bool Strategy::isApplicable() {
				return true;
			}

			void Strategy::onUpdate() {
			}

			Strategy::Strategy(const std::chrono::system_clock::duration& interval) : Loopable(interval) {
			}

			void Strategy::update() {
				// Only run the Strategy if it is applicable
				if(isApplicable()) {
					onUpdate();
				}
			}
		}
	}
}
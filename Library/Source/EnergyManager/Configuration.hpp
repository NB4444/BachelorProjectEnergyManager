#pragma once

namespace EnergyManager {
	namespace Configuration {
		/**
		 * Whether to show a warning when a loop component's loop interval has been exceeded.
		 */
		static const bool warningWhenLoopIntervalExceeded = false;

		/**
		 * Whether to show a warning when a loop component skips a frame due to a lag of 1 frame or more.
		 */
		static const bool warningWhenSkippingLoopIterations = false;
	}
}
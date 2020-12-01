#pragma once

#include <functional>

namespace EnergyManager {
	namespace Utility {
		/**
		 * Runs some code statically.
		 */
		class StaticInitializer {
		public:
			/**
			 * Creates a new StaticInitializer.
			 * @param operation The operation to execute.
			 */
			explicit StaticInitializer(const std::function<void()>& operation);
		};
	}
}
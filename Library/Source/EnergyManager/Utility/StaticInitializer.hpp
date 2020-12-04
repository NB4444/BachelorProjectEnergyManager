#pragma once

#include <functional>

namespace EnergyManager {
	namespace Utility {
		/**
		 * Runs some code statically.
		 */
		class StaticInitializer {
			/**
			 * The operation called on destruct.
			 */
			std::function<void()> destructOperation_;

		public:
			/**
			 * Creates a new StaticInitializer.
			 * @param operation The operation to execute.
			 */
			explicit StaticInitializer(
				const std::function<void()>& operation,
				std::function<void()> destructOperation = [] {
				});

			/**
			 * Destructs the initialier.
			 */
			~StaticInitializer();
		};
	}
}
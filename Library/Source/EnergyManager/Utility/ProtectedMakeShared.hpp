#pragma once

namespace EnergyManager {
	namespace Utility {
		/**
		 * See https://stackoverflow.com/a/56676533/979732.
		 * @tparam Object The Object to instantiate.
		 * @tparam Arguments The argument types.
		 * @param arguments The arguments.
		 * @return A shared pointer to the Object.
		 */
		template<typename Object, typename... Arguments>
		std::shared_ptr<Object> protectedMakeShared(Arguments&&... arguments) {
			struct Wrapper : public Object {
				Wrapper(Arguments&&... arguments) : Object { std::forward<Arguments>(arguments)... } {
				}
			};

			return std::make_shared<Wrapper>(std::forward<Arguments>(arguments)...);
		}
	}
}
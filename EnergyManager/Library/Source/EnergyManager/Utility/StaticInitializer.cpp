#include "./StaticInitializer.hpp"

namespace EnergyManager {
	namespace Utility {
		StaticInitializer::StaticInitializer(const std::function<void()>& operation) {
			operation();
		}
	}
}
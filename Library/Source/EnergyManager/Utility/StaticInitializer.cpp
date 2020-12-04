#include "./StaticInitializer.hpp"

#include <utility>

namespace EnergyManager {
	namespace Utility {
		StaticInitializer::StaticInitializer(const std::function<void()>& operation, std::function<void()> destructOperation) : destructOperation_(std::move(destructOperation)) {
			operation();
		}

		StaticInitializer::~StaticInitializer() {
			destructOperation_();
		}
	}
}
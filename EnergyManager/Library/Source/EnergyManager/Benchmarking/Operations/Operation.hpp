#pragma once

#include "EnergyManager/Utility/Runnable.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * An operation that uses dummy data to simulate a real Workload.
			 */
			class Operation : public Utility::Runnable {};
		}
	}
}
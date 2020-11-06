#pragma once

#include "EnergyManager/Benchmarking/Operations/Operation.hpp"

namespace EnergyManager {
	namespace Benchmarking {
		namespace Operations {
			/**
			 * An Operation that uses dummy data and executes on the GPU.
			 */
			class GPUOperation : public virtual Operation {};
		}
	}
}
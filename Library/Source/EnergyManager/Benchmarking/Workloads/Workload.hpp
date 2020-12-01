#pragma once

#include "EnergyManager/Benchmarking/Operations/Operation.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Utility/Runnable.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace EnergyManager {
	namespace Benchmarking {
		namespace Workloads {
			/**
			 * A workload of Operations that use dummy data to simulate a real workload.
			 */
			class Workload : public Utility::Runnable {
				/**
				 * The Operations to perform.
				 */
				std::vector<std::shared_ptr<Operations::Operation>> operations_;

			protected:
				void onRun() final;

			public:
				/**
				 * Creates a new Workload.
				 * @param operations The Operations to perform.
				 */
				explicit Workload(std::vector<std::shared_ptr<Operations::Operation>> operations = {});

				/**
				 * Adds an Operation to the Workload.
				 * @param operation The Operation to add.
				 */
				void addOperation(const std::shared_ptr<Operations::Operation>& operation);
			};
		}
	}
}
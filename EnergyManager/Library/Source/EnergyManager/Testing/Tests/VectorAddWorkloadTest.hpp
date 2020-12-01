#pragma once

#include "EnergyManager/Testing/Tests/WorkloadTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			/**
			 * Tests the VectorAddWorkload.
			 */
			class VectorAddWorkloadTest : public WorkloadTest {
			public:
				/**
				 * Creates a new VectorAddWorkloadTest from command line arguments.
				 * @param arguments The command line arguments.
				 */
				VectorAddWorkloadTest(const std::map<std::string, std::string>& arguments);
			};
		}
	}
}
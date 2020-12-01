#pragma once

#include "EnergyManager/Testing/Tests/WorkloadTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			/**
			 * Tests the AllocateFreeWorkload.
			 */
			class AllocateFreeWorkloadTest : public WorkloadTest {
			public:
				/**
				 * Creates a new AllocateFreeWorkloadTest from command line arguments.
				 * @param arguments The command line arguments.
				 */
				AllocateFreeWorkloadTest(const std::map<std::string, std::string>& arguments);
			};
		}
	}
}
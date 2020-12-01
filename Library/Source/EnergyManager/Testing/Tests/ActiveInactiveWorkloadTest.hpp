#pragma once

#include "EnergyManager/Testing/Tests/WorkloadTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			/**
			 * Tests the ActiveInactiveWorkload.
			 */
			class ActiveInactiveWorkloadTest : public WorkloadTest {
			public:
				/**
				 * Creates a new ActiveInactiveWorkloadTest from command line arguments.
				 * @param arguments The command line arguments.
				 */
				ActiveInactiveWorkloadTest(const std::map<std::string, std::string>& arguments);
			};
		}
	}
}
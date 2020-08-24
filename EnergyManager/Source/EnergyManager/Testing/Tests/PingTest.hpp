#pragma once

#include "ApplicationTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		class TestResults;

		namespace Tests {
			class PingTest :
				public ApplicationTest {
				public:
					PingTest(const std::string& name, const std::string& host, const int& times);
			};
		}
	}
}
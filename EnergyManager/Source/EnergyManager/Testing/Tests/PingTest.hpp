#pragma once

#include "ApplicationTest.hpp"

namespace EnergyManager {
	namespace Testing {
		class TestResults;

		namespace Tests {
			class PingTest : public ApplicationTest {
			public:
				PingTest(const std::string& host, const int& times);
			};
		}
	}
}
#pragma once

#include "EnergyManager/Testing/Tests/ApplicationTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Hardware {
		class CPU;
		class GPU;
	}

	namespace Testing {
		class TestResults;

		namespace Tests {
			class MatrixMultiplyTest : public ApplicationTest {
			public:
				MatrixMultiplyTest(
					const std::string& name,
					const Hardware::CPU& cpu,
					const Hardware::GPU& gpu,
					const size_t& matrixAWidth,
					const size_t& matrixAHeight,
					const size_t& matrixBWidth,
					const size_t& matrixBHeight);
			};
		}
	}
}
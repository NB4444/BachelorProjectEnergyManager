#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/MatrixMultiplyTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class FixedFrequencyMatrixMultiplyTest : public MatrixMultiplyTest {
				Hardware::CPU& cpu_;

				Hardware::GPU& gpu_;

				unsigned long minimumCPUFrequency_;

				unsigned long maximumCPUFrequency_;

				unsigned long minimumGPUFrequency_;

				unsigned long maximumGPUFrequency_;

			protected:
				std::map<std::string, std::string> onRun() override;

			public:
				FixedFrequencyMatrixMultiplyTest(
					const std::string& name,
					Hardware::CPU& cpu,
					Hardware::GPU& gpu,
					const size_t& matrixAWidth,
					const size_t& matrixAHeight,
					const size_t& matrixBWidth,
					const size_t& matrixBHeight,
					const unsigned long& minimumCPUFrequency,
					const unsigned long& maximumCPUFrequency,
					const unsigned long& minimumGPUFrequency,
					const unsigned long& maximumGPUFrequency);
			};
		}
	}
}
#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/Tests/MatrixMultiplyTest.hpp"

#include <string>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			class FixedFrequencyMatrixMultiplyTest : public MatrixMultiplyTest {
				std::shared_ptr<Hardware::CPU> cpu_;

				std::shared_ptr<Hardware::GPU> gpu_;

				unsigned long minimumCPUFrequency_;

				unsigned long maximumCPUFrequency_;

				unsigned long minimumGPUFrequency_;

				unsigned long maximumGPUFrequency_;

			protected:
				std::map<std::string, std::string> onRun() override;

			public:
				FixedFrequencyMatrixMultiplyTest(
					const std::string& name,
					const std::shared_ptr<Hardware::CPU>& cpu,
					const std::shared_ptr<Hardware::GPU>& gpu,
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
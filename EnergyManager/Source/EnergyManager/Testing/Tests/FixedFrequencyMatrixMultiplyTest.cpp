#include "./FixedFrequencyMatrixMultiplyTest.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> FixedFrequencyMatrixMultiplyTest::onRun() {
				// Set the clock rates
				// TODO: Set the CPU clock rate
				gpu_.setCoreClockRate(minimumGPUFrequency_, maximumGPUFrequency_);

				// Run the test
				auto results = MatrixMultiplyTest::onRun();

				// Reset the clock rates
				gpu_.resetCoreClockRate();

				return results;
			}

			FixedFrequencyMatrixMultiplyTest::FixedFrequencyMatrixMultiplyTest(
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
				const unsigned long& maximumGPUFrequency)
				: MatrixMultiplyTest(name, cpu, gpu, matrixAWidth, matrixAHeight, matrixBWidth, matrixBHeight)
				, cpu_(cpu)
				, gpu_(gpu)
				, minimumCPUFrequency_(minimumGPUFrequency)
				, maximumCPUFrequency_(maximumGPUFrequency)
				, minimumGPUFrequency_(minimumGPUFrequency)
				, maximumGPUFrequency_(maximumGPUFrequency) {
			}
		}
	}
}
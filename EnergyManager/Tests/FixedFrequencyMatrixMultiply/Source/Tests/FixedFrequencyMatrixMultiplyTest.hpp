#pragma once

#include <EnergyManager/Hardware/CPU.hpp>
#include <string>
#include <vector>

#include "../../MatrixMultiply/Source/Tests/MatrixMultiplyTest.hpp"

namespace Tests {
	/**
	 * Tests matrix multiplication at a fixed frequency.
	 */
	class FixedFrequencyMatrixMultiplyTest : public MatrixMultiplyTest {
		/**
		 * The CPUs to use.
		 */
		std::vector<std::shared_ptr<EnergyManager::Hardware::CPU>> cpus_;

		/**
		 * The GPU to use.
		 */
		std::shared_ptr<EnergyManager::Hardware::GPU> gpu_;

		/**
		 * The minimum CPU frequency.
		 */
		unsigned long minimumCPUFrequency_;

		/**
		 * The maximum CPU frequency.
		 */
		unsigned long maximumCPUFrequency_;

		/**
		 * The minimum GPU frequency.
		 */
		unsigned long minimumGPUFrequency_;

		/**
		 * The maximum GPU frequency.
		 */
		unsigned long maximumGPUFrequency_;

	protected:
		std::map<std::string, std::string> onTest() override;

	public:
		/**
		 * Creates a new FixedFrequencyMatrixMultiplyTest.
		 * @param name The name of the Test.
		 * @param cpus The CPUs to use.
		 * @param gpu The GPU to use.
		 * @param matrixAWidth The width of matrix A.
		 * @param matrixAHeight The height of matrix A.
		 * @param matrixBWidth The width of matrix B.
		 * @param matrixBHeight The height of matrix B.
		 * @param minimumCPUFrequency The minimum CPU frequency.
		 * @param maximumCPUFrequency The maximum CPU frequency.
		 * @param minimumGPUFrequency The minimum GPU frequency.
		 * @param maximumGPUFrequency The maximum GPU frequency.
		 * @param applicationMonitorPollingInterval The interval at which to poll that Application's status.
		 * @param monitors The Monitors to use.
		 */
		FixedFrequencyMatrixMultiplyTest(
			const std::string& name,
			std::vector<std::shared_ptr<EnergyManager::Hardware::CPU>> cpus,
			std::shared_ptr<EnergyManager::Hardware::GPU> gpu,
			const size_t& matrixAWidth,
			const size_t& matrixAHeight,
			const size_t& matrixBWidth,
			const size_t& matrixBHeight,
			const unsigned long& minimumCPUFrequency,
			const unsigned long& maximumCPUFrequency,
			const unsigned long& minimumGPUFrequency,
			const unsigned long& maximumGPUFrequency,
			const std::chrono::system_clock::duration& applicationMonitorPollingInterval,
			const std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>>& monitors = {});
	};
}

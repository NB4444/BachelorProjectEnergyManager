#include "./FixedFrequencyMatrixMultiplyTest.hpp"

#include <utility>

namespace Tests {
	std::map<std::string, std::string> FixedFrequencyMatrixMultiplyTest::onTest() {
		// Set the clock rates
		for(const auto& cpu : cpus_) {
			cpu->setCoreClockRate(minimumCPUFrequency_, maximumCPUFrequency_);
		}
		gpu_->setCoreClockRate(minimumGPUFrequency_, maximumGPUFrequency_);

		// Disable automatic boosting
		bool autoBoostedClocksEnabled;
		bool autoBoostedClocksSet = false;
		try {
			autoBoostedClocksEnabled = gpu_->getAutoBoostedClocksEnabled();
			if(autoBoostedClocksEnabled) {
				gpu_->setAutoBoostedClocksEnabled(false);
				autoBoostedClocksSet = true;
			}
		} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
		}

		// Run the test
		auto results = MatrixMultiplyTest::onTest();

		// Reset the clock rates
		for(const auto& cpu : cpus_) {
			cpu->resetCoreClockRate();
		}
		gpu_->resetCoreClockRate();

		// Re-enable automatic boosting
		if(autoBoostedClocksSet) {
			try {
				gpu_->setAutoBoostedClocksEnabled(autoBoostedClocksEnabled);
			} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
			}
		}

		return results;
	}

	FixedFrequencyMatrixMultiplyTest::FixedFrequencyMatrixMultiplyTest(
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
		const std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>>& monitors)
		: MatrixMultiplyTest(name, cpus, gpu, matrixAWidth, matrixAHeight, matrixBWidth, matrixBHeight, applicationMonitorPollingInterval, monitors)
		, cpus_(std::move(cpus))
		, gpu_(std::move(gpu))
		, minimumCPUFrequency_(minimumGPUFrequency)
		, maximumCPUFrequency_(maximumGPUFrequency)
		, minimumGPUFrequency_(minimumGPUFrequency)
		, maximumGPUFrequency_(maximumGPUFrequency) {
	}
}

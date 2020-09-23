#include "./FixedFrequencyMatrixMultiplyTest.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> FixedFrequencyMatrixMultiplyTest::onRun() {
				// Set the clock rates
				cpu_->setCoreClockRate(minimumCPUFrequency_, maximumCPUFrequency_);
				gpu_->setCoreClockRate(minimumGPUFrequency_, maximumGPUFrequency_);

				// Disable automatic boosting
				bool autoBoostedClocksEnabled;
				bool autoBoostedClocksSet = true;
				try {
					autoBoostedClocksEnabled = gpu_->getAutoBoostedClocksEnabled();
					if(autoBoostedClocksEnabled) {
						gpu_->setAutoBoostedClocksEnabled(false);
					}
				} catch(const Utility::Exceptions::Exception& exception) {
					bool autoBoostedClocksSet = false;
				}

				// Run the test
				auto results = MatrixMultiplyTest::onRun();

				// Reset the clock rates
				cpu_->resetCoreClockRate();
				gpu_->resetCoreClockRate();

				// Re-enable automatic boosting
				if(autoBoostedClocksSet) {
					try {
						gpu_->setAutoBoostedClocksEnabled(autoBoostedClocksEnabled);
					} catch(const Utility::Exceptions::Exception& exception) {
					}
				}

				return results;
			}

			FixedFrequencyMatrixMultiplyTest::FixedFrequencyMatrixMultiplyTest(
				const std::string& name,
				const std::shared_ptr<Hardware::Node>& node,
				const std::shared_ptr<Hardware::CPU>& cpu,
				const std::shared_ptr<Hardware::GPU>& gpu,
				const size_t& matrixAWidth,
				const size_t& matrixAHeight,
				const size_t& matrixBWidth,
				const size_t& matrixBHeight,
				const unsigned long& minimumCPUFrequency,
				const unsigned long& maximumCPUFrequency,
				const unsigned long& minimumGPUFrequency,
				const unsigned long& maximumGPUFrequency,
				std::chrono::system_clock::duration applicationMonitorPollingInterval,
				std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration> monitors)
				: MatrixMultiplyTest(name, node, cpu, gpu, matrixAWidth, matrixAHeight, matrixBWidth, matrixBHeight, applicationMonitorPollingInterval, monitors)
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
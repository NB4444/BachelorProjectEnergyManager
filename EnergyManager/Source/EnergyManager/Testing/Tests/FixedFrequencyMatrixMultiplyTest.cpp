#include "./FixedFrequencyMatrixMultiplyTest.hpp"

#include "EnergyManager/Utility/Exceptions/ParseException.hpp"

#include <utility>

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			std::map<std::string, std::string> FixedFrequencyMatrixMultiplyTest::onRun() {
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
				} catch(const Utility::Exceptions::Exception& exception) {
				}

				// Run the test
				auto results = MatrixMultiplyTest::onRun();

				// Reset the clock rates
				for(const auto& cpu : cpus_) {
					cpu->resetCoreClockRate();
				}
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

			void FixedFrequencyMatrixMultiplyTest::initialize() {
				Test::addParser([](const std::string& name,
								   const std::map<std::string, std::string>& parameters,
								   const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors) {
					if(name != "FixedFrequencyMatrixMultiplyTest") {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_PARSE_EXCEPTION();
					}

					return std::make_shared<EnergyManager::Testing::Tests::FixedFrequencyMatrixMultiplyTest>(
						Utility::Text::getParameter(parameters, "name"),
						Hardware::CPU::parseCPUs(Utility::Text::getParameter(parameters, "cpu")),
						EnergyManager::Hardware::GPU::getGPU(std::stoi(Utility::Text::getParameter(parameters, "gpu"))),
						std::stoi(Utility::Text::getParameter(parameters, "matrixAWidth")),
						std::stoi(Utility::Text::getParameter(parameters, "matrixAHeight")),
						std::stoi(Utility::Text::getParameter(parameters, "matrixBWidth")),
						std::stoi(Utility::Text::getParameter(parameters, "matrixBHeight")),
						std::stoul(Utility::Text::getParameter(parameters, "minimumCPUFrequency")),
						std::stoul(Utility::Text::getParameter(parameters, "maximumCPUFrequency")),
						std::stoul(Utility::Text::getParameter(parameters, "minimumGPUFrequency")),
						std::stoul(Utility::Text::getParameter(parameters, "maximumGPUFrequency")),
						std::chrono::duration_cast<std::chrono::system_clock::duration>(
							std::chrono::milliseconds(std::stoul(Utility::Text::getParameter(parameters, "applicationMonitorPollingInterval")))),
						monitors);
				});
			}

			FixedFrequencyMatrixMultiplyTest::FixedFrequencyMatrixMultiplyTest(
				const std::string& name,
				std::vector<std::shared_ptr<Hardware::CPU>> cpus,
				std::shared_ptr<Hardware::GPU> gpu,
				const size_t& matrixAWidth,
				const size_t& matrixAHeight,
				const size_t& matrixBWidth,
				const size_t& matrixBHeight,
				const unsigned long& minimumCPUFrequency,
				const unsigned long& maximumCPUFrequency,
				const unsigned long& minimumGPUFrequency,
				const unsigned long& maximumGPUFrequency,
				std::chrono::system_clock::duration applicationMonitorPollingInterval,
				const std::map<std::shared_ptr<Monitoring::Monitor>, std::chrono::system_clock::duration>& monitors)
				: MatrixMultiplyTest(name, cpus, gpu, matrixAWidth, matrixAHeight, matrixBWidth, matrixBHeight, applicationMonitorPollingInterval, monitors)
				, cpus_(std::move(cpus))
				, gpu_(std::move(gpu))
				, minimumCPUFrequency_(minimumGPUFrequency)
				, maximumCPUFrequency_(maximumGPUFrequency)
				, minimumGPUFrequency_(minimumGPUFrequency)
				, maximumGPUFrequency_(maximumGPUFrequency) {
			}
		}
	}
}
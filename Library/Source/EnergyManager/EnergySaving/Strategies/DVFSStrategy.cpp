#include "./DVFSStrategy.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <utility>

namespace EnergyManager {
	namespace EnergySaving {
		namespace Strategies {
			bool DVFSStrategy::isApplicable() {
				// TODO: Only run this when applicable
				return true;
			}

			void DVFSStrategy::onUpdate() {
				// Try to optimize the frequencies
				try {
					// Optimize the frequencies
					const auto optimalControlParameters = model_->optimize();

					// Update the frequencies
					cpu_->setCoreClockRate(optimalControlParameters.at("minimumCPUFrequency"), optimalControlParameters.at("maximumCPUFrequency"));
					gpu_->setCoreClockRate(optimalControlParameters.at("minimumGPUFrequency"), optimalControlParameters.at("maximumGPUFrequency"));
				} catch(const Utility::Exceptions::Exception& exception) {
					exception.log();
				}
			}

			DVFSStrategy::DVFSStrategy(
				std::shared_ptr<Hardware::CPU> cpu,
				std::shared_ptr<Hardware::GPU> gpu,
				const std::chrono::system_clock::duration& interval,
				const std::shared_ptr<Models::DVFSModel>& model)
				: Strategy(interval)
				, cpu_(std::move(cpu))
				, gpu_(std::move(gpu))
				, model_(model) {
			}

			DVFSStrategy::~DVFSStrategy() {
				// Reset CPU frequencies
				cpu_->resetCoreClockRate();

				// Reset GPU frequencies
				gpu_->resetCoreClockRate();
			}
		}
	}
}
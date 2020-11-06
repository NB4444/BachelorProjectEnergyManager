#include "./DVFSModel.hpp"

#include "EnergyManager/Utility/Logging.hpp"

#include <utility>

namespace EnergyManager {
	namespace EnergySaving {
		namespace Models {
			DVFSModel::DVFSModel(
				std::shared_ptr<Hardware::CPU> cpu,
				std::shared_ptr<Hardware::GPU> gpu,
				std::shared_ptr<Monitoring::Monitors::CPUMonitor> cpuMonitor,
				std::shared_ptr<Monitoring::Monitors::GPUMonitor> gpuMonitor,
				const std::vector<std::string>& dependentVariableNames,
				std::string modelPath,
				std::function<void()> profilingWorkload)
				: cpu_(std::move(cpu))
				, gpu_(std::move(gpu))
				, cpuMonitor_(std::move(cpuMonitor))
				, gpuMonitor_(std::move(gpuMonitor))
				, modelPath_(std::move(modelPath))
				, profilingWorkload_(std::move(profilingWorkload)) {
				auto fileExists = [&](const std::string& path) {
					FILE* file;
					if(file = fopen(path.c_str(), "r")) {
						fclose(file);

						return true;
					} else {
						return false;
					}
				};

				if(fileExists(modelPath_)) {
					Utility::Logging::logInformation("Found existing DVFS model, loading it...");
					linearRegression_ = Utility::MachineLearning::LinearRegression(modelPath_, dependentVariableNames);
				}
			}

			void DVFSModel::save() {
				// Save the model
				linearRegression_.save(modelPath_);
			}

			std::map<std::string, double> DVFSModel::predict(
				const Utility::Units::Hertz& minimumCPUFrequency,
				const Utility::Units::Hertz& maximumCPUFrequency,
				const Utility::Units::Hertz& minimumGPUFrequency,
				const Utility::Units::Hertz& maximumGPUFrequency) const {
				return onPredict(minimumCPUFrequency, maximumCPUFrequency, minimumGPUFrequency, maximumGPUFrequency);
			}

			std::map<std::string, double> DVFSModel::optimize() const {
				return onOptimize();
			}
		}
	}
}

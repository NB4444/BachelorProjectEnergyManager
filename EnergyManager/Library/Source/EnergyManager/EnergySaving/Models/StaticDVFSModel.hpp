#pragma once

#include "EnergyManager/EnergySaving/Models/DVFSModel.hpp"
#include "EnergyManager/Monitoring/Monitors/CPUMonitor.hpp"
#include "EnergyManager/Monitoring/Monitors/GPUMonitor.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/MachineLearning/LinearRegression.hpp"
#include "EnergyManager/Utility/Runnable.hpp"

#include <ensmallen.hpp>
#include <memory>

namespace EnergyManager {
	namespace EnergySaving {
		namespace Models {
			/**
			 * Models the parameters used by DVFS in a static manner, where the model is only used at application startup and total work is known in advance.
			 */
			class StaticDVFSModel : public DVFSModel {
				/**
				 * Keeps track of the optimal values found during training.
				 */
				std::map<std::string, double> getOptimalValues() const;

			protected:
				void onRun() override;

				std::map<std::string, double> onPredict(
					const Utility::Units::Hertz& minimumCPUFrequency,
					const Utility::Units::Hertz& maximumCPUFrequency,
					const Utility::Units::Hertz& minimumGPUFrequency,
					const Utility::Units::Hertz& maximumGPUFrequency) const override;

				std::map<std::string, double> onOptimize() const override;

			public:
				/**
				 * Creates a new StaticDVFSModel.
				 * @param cpu The CPU.
				 * @param gpu The GPU.
				 * @param cpuMonitor The CPUMonitor to use.
				 * @param gpuMonitor The GPUMonitor to use.
				 * @param modelPath The path to the model to use to tune the parameters.
				 * @param profilingWorkload The workload to run during profiling.
				 * @param profilingRuns The amount of profiling runs to do.
				 */
				StaticDVFSModel(
					const std::shared_ptr<Hardware::CPU>& cpu,
					const std::shared_ptr<Hardware::GPU>& gpu,
					const std::shared_ptr<Monitoring::Monitors::CPUMonitor>& cpuMonitor,
					const std::shared_ptr<Monitoring::Monitors::GPUMonitor>& gpuMonitor,
					const std::string& modelPath = "",
					const std::function<void()>& profilingWorkload = [] {
					});
			};
		}
	}
}
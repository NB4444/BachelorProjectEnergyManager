#pragma once

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
			 * Models the parameters used by DVFS.
			 */
			class DVFSModel : public Utility::Runnable {
			protected:
				/**
				 * The model that is used to predict parameters.
				 */
				Utility::MachineLearning::LinearRegression linearRegression_;

				/**
				 * The path to the model file.
				 */
				std::string modelPath_;

				/**
				 * The CPU.
				 */
				std::shared_ptr<Hardware::CPU> cpu_;

				/**
				 * The GPU.
				 */
				std::shared_ptr<Hardware::GPU> gpu_;

				/**
				 * The monitor used to monitor the CPU.
				 */
				std::shared_ptr<Monitoring::Monitors::CPUMonitor> cpuMonitor_;

				/**
				 * The monitor used to monitor the GPU.
				 */
				std::shared_ptr<Monitoring::Monitors::GPUMonitor> gpuMonitor_;

				/**
				 * The workload to run during profiling.
				 */
				std::function<void()> profilingWorkload_;

				/**
				 * Predicts the amount of Joules per FLOP for the current period.
				 * @param minimumCPUFrequency The minimum CPU frequency.
				 * @param maximumCPUFrequency The maximum CPU frequency.
				 * @param minimumGPUFrequency The minimum GPU frequency.
				 * @param maximumGPUFrequency The maximum GPU frequency.
				 * @return The amount of Joules per FLOP.
				 */
				virtual std::map<std::string, double> onPredict(
					const Utility::Units::Hertz& minimumCPUFrequency,
					const Utility::Units::Hertz& maximumCPUFrequency,
					const Utility::Units::Hertz& minimumGPUFrequency,
					const Utility::Units::Hertz& maximumGPUFrequency) const = 0;

				/**
				 * Determines the optimal parameters to achieve the lowest amount of Joules per FLOP given the current performance variables.
				 * @return The optimal parameters.
				 */
				virtual std::map<std::string, double> onOptimize() const = 0;

			public:
				/**
				 * Creates a new DVFSModel.
				 * @param cpu The CPU.
				 * @param gpu The GPU.
				 * @param cpuMonitor The CPUMonitor to use.
				 * @param gpuMonitor The GPUMonitor to use.
				 * @param modelPath The path to the model to use to tune the parameters.
				 * @param profilingWorkload The workload to run during profiling.
				 * @param profilingRuns The amount of profiling runs to do.
				 */
				DVFSModel(
					std::shared_ptr<Hardware::CPU> cpu,
					std::shared_ptr<Hardware::GPU> gpu,
					std::shared_ptr<Monitoring::Monitors::CPUMonitor> cpuMonitor,
					std::shared_ptr<Monitoring::Monitors::GPUMonitor> gpuMonitor,
					const std::vector<std::string>& dependentVariableNames,
					std::string modelPath = "",
					std::function<void()> profilingWorkload = [] {
					});

				/**
				 * Saves the model.
				 */
				void save();

				/**
				 * Predicts the amount of Joules per FLOP for the current period.
				 * @param minimumCPUFrequency The minimum CPU frequency.
				 * @param maximumCPUFrequency The maximum CPU frequency.
				 * @param minimumGPUFrequency The minimum GPU frequency.
				 * @param maximumGPUFrequency The maximum GPU frequency.
				 * @return The amount of Joules per FLOP.
				 */
				std::map<std::string, double> predict(
					const Utility::Units::Hertz& minimumCPUFrequency,
					const Utility::Units::Hertz& maximumCPUFrequency,
					const Utility::Units::Hertz& minimumGPUFrequency,
					const Utility::Units::Hertz& maximumGPUFrequency) const;

				/**
				 * Determines the optimal parameters to achieve the lowest amount of Joules per FLOP given the current performance variables.
				 * @return The optimal parameters.
				 */
				std::map<std::string, double> optimize() const;
			};
		}
	}
}
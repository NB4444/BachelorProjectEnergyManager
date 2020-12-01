//#pragma once
//
//#include "EnergyManager/EnergySaving/Models/DVFSModel.hpp"
//#include "EnergyManager/Monitoring/CPUMonitor.hpp"
//#include "EnergyManager/Monitoring/GPUMonitor.hpp"
//#include "EnergyManager/Utility/Exceptions/Exception.hpp"
//#include "EnergyManager/Utility/MachineLearning/LinearRegression.hpp"
//#include "EnergyManager/Utility/Runnable.hpp"
//
//#include <ensmallen.hpp>
//#include <memory>
//
//namespace EnergyManager {
//	namespace EnergySaving {
//		namespace Models {
//			/**
//			 * Models the parameters used by DVFS in a dynamic manner, where the model can be used in a dynamic context where the total amount of work is not known in advance.
//			 */
//			class DynamicDVFSModel : public DVFSModel {
//				/**
//				 * The period over which to optimize the model.
//				 * This is the interval that is used for selecting training data items.
//				 */
//				std::chrono::system_clock::duration optimizationPeriod_;
//
//				/**
//				 * The end timestamp of the last optimization period.
//				 */
//				std::chrono::system_clock::time_point lastOptimizationPeriodEndTimestamp_ = std::chrono::system_clock::now();
//
//			protected:
//				void onRun() final;
//
//				std::map<std::string, double> onPredict(
//					const Utility::Units::Hertz& minimumCPUFrequency,
//					const Utility::Units::Hertz& maximumCPUFrequency,
//					const Utility::Units::Hertz& minimumGPUFrequency,
//					const Utility::Units::Hertz& maximumGPUFrequency) const final;
//
//				std::map<std::string, double> onOptimize() const final;
//
//			public:
//				/**
//				 * Creates a new DynamicDVFSModel.
//				 * @param cpu The CPU.
//				 * @param gpu The GPU.
//				 * @param cpuMonitor The CPUMonitor to use.
//				 * @param gpuMonitor The GPUMonitor to use.
//				 * @param optimizationPeriod The period over which to optimize.
//				 * @param modelPath The path to the model to use to tune the parameters.
//				 * @param profilingWorkload The workload to run during profiling.
//				 * @param profilingRuns The amount of profiling runs to do.
//				 */
//				DynamicDVFSModel(
//					const std::shared_ptr<Hardware::CPU>& cpu,
//					const std::shared_ptr<Hardware::GPU>& gpu,
//					const std::shared_ptr<Monitoring::CPUMonitor>& cpuMonitor,
//					const std::shared_ptr<Monitoring::GPUMonitor>& gpuMonitor,
//					const std::chrono::system_clock::duration& optimizationPeriod,
//					const std::string& modelPath = "",
//					const std::function<void()>& profilingWorkload =
//						[] {
//						});
//			};
//		}
//	}
//}
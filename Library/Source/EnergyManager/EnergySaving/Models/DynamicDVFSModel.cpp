//#include "./DynamicDVFSModel.hpp"
//
//#include "EnergyManager/Utility/Logging.hpp"
//#include "EnergyManager/Utility/Text.hpp"
//
//namespace EnergyManager {
//	namespace EnergySaving {
//		namespace Models {
//			void DynamicDVFSModel::onRun() {
//				// TODO: Test this function and this class, this is currently untested code!
//				const auto now = std::chrono::system_clock::now();
//
//				// Generate training data
//				std::map<std::map<std::string, double>, std::map<std::string, double>> trainingData = {};
//
//				// Loop optimization periods between now and the latest processed period
//				// TODO: Change this so that it processes every period since the start
//				auto currentOptimizationPeriodStart = lastOptimizationPeriodEndTimestamp_;
//				auto currentOptimizationPeriodEnd = currentOptimizationPeriodStart + optimizationPeriod_;
//				auto previousOptimizationPeriodEnd = currentOptimizationPeriodStart;
//				auto previousOptimizationPeriodStart = previousOptimizationPeriodEnd - optimizationPeriod_;
//				while(currentOptimizationPeriodEnd < std::chrono::system_clock::now()) {
//					// Ensure that we only process data after we have had at least one full optimization period
//					if(cpuMonitor_->hasVariableValues() && gpuMonitor_->hasVariableValues()
//					   && (previousOptimizationPeriodEnd > cpuMonitor_->getStartTimestamp() || previousOptimizationPeriodEnd > gpuMonitor_->getStartTimestamp())) {
//						// Determine performance values over the previous period
//						const auto previousCPUFlops = cpuMonitor_->calculateDifference("flops", previousOptimizationPeriodStart, previousOptimizationPeriodEnd);
//						const auto previousGPUFlops = gpuMonitor_->calculateDifference("flops", previousOptimizationPeriodStart, previousOptimizationPeriodEnd);
//
//						// Determine control variable values over the previous period
//						const auto minimumCPUFrequency = cpuMonitor_->calculateMinimum("coreClockRate", previousOptimizationPeriodStart, previousOptimizationPeriodEnd);
//						const auto maximumCPUFrequency = cpuMonitor_->calculateMaximum("coreClockRate", previousOptimizationPeriodStart, previousOptimizationPeriodEnd);
//						const auto minimumGPUFrequency = gpuMonitor_->calculateMinimum("coreClockRate", previousOptimizationPeriodStart, previousOptimizationPeriodEnd);
//						const auto maximumGPUFrequency = gpuMonitor_->calculateMaximum("coreClockRate", previousOptimizationPeriodStart, previousOptimizationPeriodEnd);
//
//						// Determine response variables over the current period
//						const auto currentCPUFLOPs = cpuMonitor_->calculateDifference("flops", currentOptimizationPeriodStart, currentOptimizationPeriodEnd);
//						const auto currentGPUFLOPs = gpuMonitor_->calculateDifference("flops", currentOptimizationPeriodStart, currentOptimizationPeriodEnd);
//						const auto currentFLOPs = currentCPUFLOPs + currentGPUFLOPs;
//						const auto cpuJoules = cpuMonitor_->calculateDifference("energyConsumption", currentOptimizationPeriodStart, currentOptimizationPeriodEnd);
//						const auto gpuJoules = gpuMonitor_->calculateDifference("energyConsumption", currentOptimizationPeriodStart, currentOptimizationPeriodEnd);
//						// TODO: Optimize for node power consumption instead
//						const auto energyConsumption = cpuJoules + gpuJoules;
//
//						// Update optimal values
//						if(optimalValues_.find("energyConsumption") == optimalValues_.end() || optimalValues_["energyConsumption"] > energyConsumption) {
//							optimalValues_["energyConsumption"] = energyConsumption;
//						}
//						if(optimalValues_.find("flops") == optimalValues_.end() || optimalValues_["flops"] < currentFLOPs) {
//							optimalValues_["flops"] = currentFLOPs;
//						}
//
//						// Add the training data
//						trainingData[{ { "cpuFLOPs", previousCPUFlops },
//									   { "minimumCPUFrequency", minimumCPUFrequency },
//									   { "maximumCPUFrequency", maximumCPUFrequency },
//									   { "gpuFLOPs", previousGPUFlops },
//									   { "minimumGPUFrequency", minimumGPUFrequency },
//									   { "maximumGPUFrequency", maximumGPUFrequency } }]
//							= { { "energyConsumption", energyConsumption }, { "flops", currentFLOPs } };
//
//						// Go to the next period
//						currentOptimizationPeriodStart = currentOptimizationPeriodEnd;
//						currentOptimizationPeriodEnd = currentOptimizationPeriodStart + optimizationPeriod_;
//					}
//				}
//
//				//return trainingData;
//			}
//
//			std::map<std::string, double> DynamicDVFSModel::onPredict(
//				const Utility::Units::Hertz& minimumCPUFrequency,
//				const Utility::Units::Hertz& maximumCPUFrequency,
//				const Utility::Units::Hertz& minimumGPUFrequency,
//				const Utility::Units::Hertz& maximumGPUFrequency) const {
//				const auto lastOptimizationPeriodEnd = std::chrono::system_clock::now();
//				const auto lastOptimizationPeriodStart = lastOptimizationPeriodEnd - optimizationPeriod_;
//
//				// Check if there is data
//				if(!(cpuMonitor_->hasVariableValues() && gpuMonitor_->hasVariableValues()
//					 && (lastOptimizationPeriodEnd > cpuMonitor_->getStartTimestamp() || lastOptimizationPeriodEnd > gpuMonitor_->getStartTimestamp()))) {
//					ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("No data to predict with");
//				}
//
//				// Determine performance values over last period
//				const auto cpuFLOPs = cpuMonitor_->calculateDifference("flops", lastOptimizationPeriodStart, lastOptimizationPeriodEnd);
//				const auto gpuFLOPs = gpuMonitor_->calculateDifference("flops", lastOptimizationPeriodStart, lastOptimizationPeriodEnd);
//
//				// Do the prediction
//				return linearRegression_.predict(std::map<std::string, double> { { "cpuFLOPs", cpuFLOPs },
//																				 { "minimumCPUFrequency", minimumCPUFrequency.toValue() },
//																				 { "maximumCPUFrequency", maximumCPUFrequency.toValue() },
//																				 { "gpuFLOPs", gpuFLOPs },
//																				 { "minimumGPUFrequency", minimumGPUFrequency.toValue() },
//																				 { "maximumGPUFrequency", maximumGPUFrequency.toValue() } });
//			}
//
//			std::map<std::string, double> DynamicDVFSModel::onOptimize() const {
//				class LinearRegressionFunction {
//					const DynamicDVFSModel& dvfsModel_;
//
//				public:
//					explicit LinearRegressionFunction(const DynamicDVFSModel& continuousDVFSModel) : dvfsModel_(continuousDVFSModel) {
//					}
//
//					double Evaluate(const arma::mat& independentVariables) const {
//						const auto& minimumCPUFrequency = independentVariables(0, 0);
//						const auto& maximumCPUFrequency = independentVariables(0, 1);
//						const auto& minimumGPUFrequency = independentVariables(0, 2);
//						const auto& maximumGPUFrequency = independentVariables(0, 3);
//
//						// Predict control variable values
//						const auto prediction = dvfsModel_.predict(minimumCPUFrequency, maximumCPUFrequency, minimumGPUFrequency, maximumGPUFrequency);
//
//						// Ensure that all values are within range
//						const auto minimumCPUFrequencyBound = dvfsModel_.cpu_->getMinimumCoreClockRate();
//						const auto maximumCPUFrequencyBound = dvfsModel_.cpu_->getMaximumCoreClockRate();
//						const auto minimumGPUFrequencyBound = 1;
//						const auto maximumGPUFrequencyBound = dvfsModel_.gpu_->getMaximumCoreClockRate();
//						if(minimumCPUFrequency < minimumCPUFrequencyBound || maximumCPUFrequency > maximumCPUFrequencyBound || minimumGPUFrequency < minimumGPUFrequencyBound
//						   || maximumGPUFrequency > maximumGPUFrequencyBound) {
//							// Out of bounds
//							return std::numeric_limits<double>::infinity();
//						}
//
//						// Return score
//						// Minimize watts per flop and runtime
//						// The values are normalized by dividing them by the optimal values found during training
//						return prediction.at("energyConsumption") / dvfsModel_.optimalValues_.at("energyConsumption") - prediction.at("flops") / dvfsModel_.optimalValues_.at("flops");
//					}
//				};
//				LinearRegressionFunction linearRegressionFunction(*this);
//
//				// Get the initial point
//				arma::mat optimizedPoint(
//					Utility::Text::toString(cpu_->getMinimumCoreClockRate().toValue()) + " " + Utility::Text::toString(cpu_->getMaximumCoreClockRate().toValue()) + " 1 "
//					+ Utility::Text::toString(gpu_->getMaximumCoreClockRate().toValue()));
//
//				// Optimize
//				auto schedule = ens::ExponentialSchedule();
//				// TODO: Change SA parameters to find the best values
//				ens::SA<> optimizer(schedule, 1000000, 10000.0, 1000, 100, 1e-5, 3, 20, 1.0, 1.0);
//				optimizer.Optimize(linearRegressionFunction, optimizedPoint);
//
//				// Return the results
//				return { { "minimumCPUFrequency", optimizedPoint(0, 0) },
//						 { "maximumCPUFrequency", optimizedPoint(0, 1) },
//						 { "minimumGPUFrequency", optimizedPoint(0, 2) },
//						 { "maximumGPUFrequency", optimizedPoint(0, 3) } };
//			}
//
//			DynamicDVFSModel::DynamicDVFSModel(
//				const std::shared_ptr<Hardware::CPU>& cpu,
//				const std::shared_ptr<Hardware::GPU>& gpu,
//				const std::shared_ptr<Monitoring::CPUMonitor>& cpuMonitor,
//				const std::shared_ptr<Monitoring::GPUMonitor>& gpuMonitor,
//				const std::chrono::system_clock::duration& optimizationPeriod,
//				const std::string& modelPath,
//				const std::function<void()>& profilingWorkload)
//				: DVFSModel(cpu, gpu, cpuMonitor, gpuMonitor, { "energyConsumption", "flops" }, modelPath, profilingWorkload)
//				, optimizationPeriod_(optimizationPeriod) {
//			}
//		}
//	}
//}

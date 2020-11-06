#include "./StaticDVFSModel.hpp"

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

namespace EnergyManager {
	namespace EnergySaving {
		namespace Models {
			std::map<std::string, double> StaticDVFSModel::getOptimalValues() const {
				double energyConsumption;
				bool energyConsumptionMeasured = false;
				double runtime;
				bool runtimeMeasured = false;

				for(const auto& trainingDatum : linearRegression_.getTrainingData()) {
					double currentEnergyConsumption = trainingDatum.second.at("energyConsumption");
					if(!energyConsumptionMeasured || currentEnergyConsumption < energyConsumption) {
						energyConsumption = currentEnergyConsumption;
					}
					double currentRuntime = trainingDatum.second.at("runtime");
					if(!runtimeMeasured || currentRuntime < runtime) {
						runtime = currentRuntime;
					}
				}

				return { { "energyConsumption", energyConsumption }, { "runtime", runtime } };
			}

			void StaticDVFSModel::onRun() {
				std::vector<std::pair<std::map<std::string, double>, std::map<std::string, double>>> trainingData;

				// Calculate slices per device
				const auto cpuSlices = 5;
				const auto gpuSlices = 5;
				const auto cpuSlice = std::abs(static_cast<long>(cpu_->getMinimumCoreClockRate().toValue()) - static_cast<long>(cpu_->getMaximumCoreClockRate().toValue())) / cpuSlices;
				const auto gpuSlice = std::abs(static_cast<long>(gpu_->getMaximumCoreClockRate().toValue() / 10) - static_cast<long>(gpu_->getMaximumCoreClockRate().toValue())) / gpuSlices;

				// Do the profiling runs
				Utility::Logging::logInformation("Generating profile parameters...");
				std::vector<std::tuple<unsigned long, unsigned long, unsigned long, unsigned long>> runParameters;

				unsigned long minimumCPUFrequency = cpu_->getMinimumCoreClockRate().toValue();
				unsigned long maximumCPUFrequency = minimumCPUFrequency + cpuSlice;
				while(maximumCPUFrequency <= cpu_->getMaximumCoreClockRate()) {
					unsigned long minimumGPUFrequency = gpu_->getMaximumCoreClockRate().toValue() / 10;
					unsigned long maximumGPUFrequency = minimumGPUFrequency + gpuSlice;

					while(maximumGPUFrequency <= gpu_->getMaximumCoreClockRate()) {
						runParameters.push_back(std::make_tuple(minimumCPUFrequency, maximumCPUFrequency, minimumGPUFrequency, maximumGPUFrequency));

						minimumGPUFrequency = maximumGPUFrequency;
						maximumGPUFrequency = minimumGPUFrequency + gpuSlice;
					}

					minimumCPUFrequency = maximumCPUFrequency;
					maximumCPUFrequency = minimumCPUFrequency + cpuSlice;
				}

				Utility::Logging::logInformation("Profiling model...");
				for(unsigned int runIndex = 0; runIndex < runParameters.size(); ++runIndex) {
					const auto& runParameter = runParameters[runIndex];

					Utility::Logging::logInformation("Running profiling run %d / %d...", runIndex + 1, runParameters.size());

					// Start the monitor threads
					std::vector<std::shared_ptr<Monitoring::Monitors::Monitor>> monitors = { cpuMonitor_, gpuMonitor_ };
					for(auto& monitor : monitors) {
						Utility::Logging::logInformation("Starting monitor %s thread...", monitor->getName().c_str());
						monitor->reset(); // Ensure that there is no previous data
						monitor->run(true);
					}

					// Set the parameters
					const auto minimumCPUFrequency = std::get<0>(runParameter);
					const auto maximumCPUFrequency = std::get<1>(runParameter);
					Utility::Logging::logInformation("Setting CPU frequency range to [%lu, %lu]...", minimumCPUFrequency, maximumCPUFrequency);
					cpu_->setCoreClockRate(minimumCPUFrequency, maximumCPUFrequency);

					const auto minimumGPUFrequency = std::get<2>(runParameter);
					const auto maximumGPUFrequency = std::get<3>(runParameter);
					Utility::Logging::logInformation("Setting GPU frequency range to [%lu, %lu]...", minimumGPUFrequency, maximumGPUFrequency);
					gpu_->setCoreClockRate(minimumGPUFrequency, maximumGPUFrequency);

					// Run the application and collect data
					Utility::Logging::logInformation("Running workload...");
					profilingWorkload_();

					// Stop all monitors threads
					for(auto& monitor : monitors) {
						Utility::Logging::logInformation("Stopping monitor %s...", monitor->getName().c_str());
						monitor->stop(true);
					}

					// Ensure that we only process data after we have had at least some data
					if(!(cpuMonitor_->hasVariableValues() && gpuMonitor_->hasVariableValues())) {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("No data to train on");
					}

					const auto now = std::chrono::system_clock::now();

					// Determine response variables over the current period
					const auto cpuJoules = cpuMonitor_->calculateDifference("energyConsumption");
					const auto gpuJoules = gpuMonitor_->calculateDifference("energyConsumption");
					// TODO: Optimize for node power consumption instead
					const auto energyConsumption = cpuJoules + gpuJoules;
					const auto runtime = cpuMonitor_->calculateDifference("runtime");

					// Generate training data
					trainingData.push_back({ { { "minimumCPUFrequency", minimumCPUFrequency },
											   { "maximumCPUFrequency", maximumCPUFrequency },
											   { "minimumGPUFrequency", minimumGPUFrequency },
											   { "maximumGPUFrequency", maximumGPUFrequency } },
											 { { "energyConsumption", energyConsumption }, { "runtime", runtime } } });
				}

				// Train the model
				Utility::Logging::logInformation("Training DVFS model...");
				const auto optimalValues = getOptimalValues();
				Utility::Logging::logInformation("Found optimal energy consumption of %f", optimalValues.at("energyConsumption"));
				Utility::Logging::logInformation("Found optimal runtime of %f", optimalValues.at("runtime"));
				linearRegression_.train(trainingData);
			}

			std::map<std::string, double> StaticDVFSModel::onPredict(
				const Utility::Units::Hertz& minimumCPUFrequency,
				const Utility::Units::Hertz& maximumCPUFrequency,
				const Utility::Units::Hertz& minimumGPUFrequency,
				const Utility::Units::Hertz& maximumGPUFrequency) const {
				// Do the prediction
				return linearRegression_.predict(std::map<std::string, double> { { "minimumCPUFrequency", minimumCPUFrequency.toValue() },
																				 { "maximumCPUFrequency", maximumCPUFrequency.toValue() },
																				 { "minimumGPUFrequency", minimumGPUFrequency.toValue() },
																				 { "maximumGPUFrequency", maximumGPUFrequency.toValue() } });
			}

			std::map<std::string, double> StaticDVFSModel::onOptimize() const {
				Utility::Logging::logInformation("Optimizing...");

				class LinearRegressionFunction {
					const StaticDVFSModel& dvfsModel_;

					const Utility::Units::Hertz minimumCPUFrequencyBound = dvfsModel_.cpu_->getMinimumCoreClockRate();

					const Utility::Units::Hertz maximumCPUFrequencyBound = dvfsModel_.cpu_->getMaximumCoreClockRate();

					const Utility::Units::Hertz minimumGPUFrequencyBound = 1;

					const Utility::Units::Hertz maximumGPUFrequencyBound = dvfsModel_.gpu_->getMaximumCoreClockRate();

				public:
					explicit LinearRegressionFunction(const StaticDVFSModel& periodicDVFSModel) : dvfsModel_(periodicDVFSModel) {
					}

					double Evaluate(const arma::mat& independentVariables) const {
						const auto& minimumCPUFrequency = independentVariables(0, 0);
						const auto& maximumCPUFrequency = independentVariables(0, 1);
						const auto& minimumGPUFrequency = independentVariables(0, 2);
						const auto& maximumGPUFrequency = independentVariables(0, 3);

						// Predict control variable values
						const auto prediction = dvfsModel_.predict(minimumCPUFrequency, maximumCPUFrequency, minimumGPUFrequency, maximumGPUFrequency);
						const auto energyConsumption = prediction.at("energyConsumption");
						const auto runtime = prediction.at("runtime");

						// Calculate penalty
						double penalty = 0;
						if(minimumCPUFrequency < minimumCPUFrequencyBound.toValue()) {
							penalty += 100 + std::abs(minimumCPUFrequency - minimumCPUFrequencyBound.toValue());
						}
						if(minimumCPUFrequency > maximumCPUFrequency) {
							penalty += 100 + std::abs(minimumCPUFrequency - maximumCPUFrequency);
						}
						if(maximumCPUFrequency > maximumCPUFrequencyBound.toValue()) {
							penalty += 100 + std::abs(maximumCPUFrequency - maximumCPUFrequencyBound.toValue());
						}
						if(maximumCPUFrequency < minimumCPUFrequency) {
							penalty += 100 + std::abs(maximumCPUFrequency - minimumCPUFrequency);
						}
						if(minimumGPUFrequency < minimumGPUFrequencyBound.toValue()) {
							penalty += 100 + std::abs(minimumGPUFrequency - minimumGPUFrequencyBound.toValue());
						}
						if(minimumGPUFrequency > maximumGPUFrequency) {
							penalty += 100 + std::abs(minimumGPUFrequency - maximumGPUFrequency);
						}
						if(maximumGPUFrequency > maximumGPUFrequencyBound.toValue()) {
							penalty += 100 + std::abs(maximumGPUFrequency - maximumGPUFrequencyBound.toValue());
						}
						if(maximumGPUFrequency < minimumGPUFrequency) {
							penalty += 100 + std::abs(maximumGPUFrequency - minimumGPUFrequency);
						}
						if(energyConsumption < 0) {
							penalty += 100 + std::abs(energyConsumption);
						}
						if(runtime < 0) {
							penalty += 100 + std::abs(runtime);
						}

						// Return score
						// Minimize watts per flop and runtime
						// The values are normalized by dividing them by the optimal values found during training
						const auto optimalValues = dvfsModel_.getOptimalValues();
						return energyConsumption / optimalValues.at("energyConsumption") + runtime / optimalValues.at("runtime") + penalty;
					}
				};
				LinearRegressionFunction linearRegressionFunction(*this);

				// Get the initial point
				arma::mat optimizedPoint(
					std::to_string(cpu_->getMinimumCoreClockRate().toValue()) + " " + std::to_string(cpu_->getMaximumCoreClockRate().toValue()) + " 1 "
					+ std::to_string(gpu_->getMaximumCoreClockRate().toValue()));

				// Optimize
				auto schedule = ens::ExponentialSchedule();
				// TODO: Change SA parameters to find the best values
				ens::SA<> optimizer(
					schedule,
					1000000,
					10000,
					1000,
					100,
					1e-5,
					3,
					0.1 * std::min(cpu_->getMaximumCoreClockRate().toValue(), gpu_->getMaximumCoreClockRate().toValue()),
					0.01 * std::min(cpu_->getMaximumCoreClockRate().toValue(), gpu_->getMaximumCoreClockRate().toValue()),
					5);
				optimizer.Optimize(linearRegressionFunction, optimizedPoint);

				// Extract the predicted values
				const Utility::Units::Hertz minimumCPUFrequency(optimizedPoint(0, 0));
				const Utility::Units::Hertz maximumCPUFrequency(optimizedPoint(0, 1));
				Utility::Logging::logInformation("Found optimal CPU frequency range of [%lu, %lu]", minimumCPUFrequency.toValue(), maximumCPUFrequency.toValue());

				const Utility::Units::Hertz minimumGPUFrequency(optimizedPoint(0, 2));
				const Utility::Units::Hertz maximumGPUFrequency(optimizedPoint(0, 3));
				Utility::Logging::logInformation("Found optimal GPU frequency range of [%lu, %lu]", minimumGPUFrequency.toValue(), maximumGPUFrequency.toValue());

				const auto predicted = predict(minimumCPUFrequency, maximumCPUFrequency, minimumGPUFrequency, maximumGPUFrequency);
				Utility::Logging::logInformation("Optimal frequencies predict energy consumption of %f", predicted.at("energyConsumption"));
				Utility::Logging::logInformation("Optimal frequencies predict runtime of %f", predicted.at("runtime"));

				// Return the results
				return { { "minimumCPUFrequency", minimumCPUFrequency.toValue() },
						 { "maximumCPUFrequency", maximumCPUFrequency.toValue() },
						 { "minimumGPUFrequency", minimumGPUFrequency.toValue() },
						 { "maximumGPUFrequency", maximumGPUFrequency.toValue() } };
			}

			StaticDVFSModel::StaticDVFSModel(
				const std::shared_ptr<Hardware::CPU>& cpu,
				const std::shared_ptr<Hardware::GPU>& gpu,
				const std::shared_ptr<Monitoring::Monitors::CPUMonitor>& cpuMonitor,
				const std::shared_ptr<Monitoring::Monitors::GPUMonitor>& gpuMonitor,
				const std::string& modelPath,
				const std::function<void()>& profilingWorkload)
				: DVFSModel(cpu, gpu, cpuMonitor, gpuMonitor, { "energyConsumption", "runtime" }, modelPath, profilingWorkload) {
			}
		}
	}
}

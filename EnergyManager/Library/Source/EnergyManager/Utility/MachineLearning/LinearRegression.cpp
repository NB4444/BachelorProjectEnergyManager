#include "./LinearRegression.hpp"

#include <EnergyManager/Utility/Logging.hpp>
#include <ensmallen.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>
#include <utility>

/**
 * Linear regression tutorials:
 * - https://mlpack.org/doc/mlpack-git/doxygen/lrtutorial.html
 * - https://mlpack.org/doc/mlpack-git/doxygen/lrtutorial.html#linreg_lrtut
 * - https://www.youtube.com/watch?v=O_AlbzguUQQ
 * - https://cppsecrets.com/users/489510710111510497118107979811497495464103109971051084699111109/C00-MLPACK-LinearRegression.php
 */

namespace EnergyManager {
	namespace Utility {
		namespace MachineLearning {
			arma::mat LinearRegression::getMatrix(const std::vector<std::map<std::string, double>>& data, const double& defaultValue) {
				// Collect all variable names
				const auto variables = getVariableNames(data);
				const std::vector<std::string> indexedVariables(variables.begin(), variables.end());

				// Generate the predictor variables matrix and response matrix
				arma::mat matrix(indexedVariables.size(), data.size());
				for(unsigned int row = 0; row < data.size(); ++row) {
					const auto& currentVariables = data[row];

					// Set the independent variables
					for(unsigned int column = 0; column < indexedVariables.size(); ++column) {
						const auto& variable = indexedVariables[column];

						// Matrix is column-first in mlpack
						matrix(column, row) = currentVariables.find(variable) == currentVariables.end() ? defaultValue : currentVariables.at(variable);
					}
				}

				return matrix;
			}

			arma::mat LinearRegression::getIndependentVariableMatrix(const std::map<std::map<std::string, double>, std::map<std::string, double>>& data) {
				std::vector<std::map<std::string, double>> independentVariables = {};
				std::transform(data.begin(), data.end(), std::back_inserter(independentVariables), [](const auto& item) {
					return item.first;
				});

				return getMatrix(independentVariables);
			}

			arma::mat LinearRegression::getDependentVariableMatrix(const std::map<std::map<std::string, double>, std::map<std::string, double>>& data) {
				std::vector<std::map<std::string, double>> dependentVariables = {};
				std::transform(data.begin(), data.end(), std::back_inserter(dependentVariables), [](const auto& item) {
					return item.second;
				});

				return getMatrix(dependentVariables);
			}

			std::set<std::string> LinearRegression::getVariableNames(const std::vector<std::map<std::string, double>>& data) {
				std::set<std::string> variables = {};

				for(const auto& currentData : data) {
					for(const auto& variable : currentData) {
						variables.insert(variable.first);
					}
				}

				return variables;
			}

			std::set<std::string> LinearRegression::getIndependentVariableNames(const std::map<std::map<std::string, double>, std::map<std::string, double>>& data) {
				std::vector<std::map<std::string, double>> independentVariables = {};
				std::transform(data.begin(), data.end(), std::back_inserter(independentVariables), [](const auto& item) {
					return item.first;
				});

				return getVariableNames(independentVariables);
			}

			std::set<std::string> LinearRegression::getDependentVariableNames(const std::map<std::map<std::string, double>, std::map<std::string, double>>& data) {
				std::vector<std::map<std::string, double>> dependentVariables = {};
				std::transform(data.begin(), data.end(), std::back_inserter(dependentVariables), [](const auto& item) {
					return item.second;
				});

				return getVariableNames(dependentVariables);
			}

			LinearRegression::LinearRegression(const std::string& filePath, const std::vector<std::string>& dependentVariableNames) {
				// Load training data
				logDebug("Loading training data...");
				std::ifstream ifstream(filePath);
				boost::archive::text_iarchive iarchive(ifstream);
				iarchive >> trainingData_;
				logDebug("Loaded %d training entries", trainingData_.size());

				// Load the models
				for(const auto& dependentVariableName : dependentVariableNames) {
					logDebug("Loading dependent variable model %s...", dependentVariableName.c_str());
					dependentVariableModels_[dependentVariableName] = mlpack::regression::LinearRegression();
					mlpack::data::Load(filePath + "-" + dependentVariableName + ".txt", "lr_model", dependentVariableModels_[dependentVariableName]);
				}
			}

			std::map<std::map<std::string, double>, std::map<std::string, double>> LinearRegression::getTrainingData() const {
				return trainingData_;
			}

			std::set<std::string> LinearRegression::getIndependentVariableNames() const {
				return getIndependentVariableNames(trainingData_);
			}

			std::set<std::string> LinearRegression::getDependentVariableNames() const {
				return getDependentVariableNames(trainingData_);
			}

			void LinearRegression::reset() {
				trainingData_.clear();
				dependentVariableModels_.clear();
			}

			void LinearRegression::train(const std::vector<std::pair<std::map<std::string, double>, std::map<std::string, double>>>& trainingData) {
				// Add the new data to the training set
				for(const auto& currentTrainingData : trainingData) {
					trainingData_[currentTrainingData.first] = currentTrainingData.second;
				}

				// Generate the predictor variables matrix and response matrix
				const auto predictors = getIndependentVariableMatrix(trainingData_);
				const auto responses = getDependentVariableMatrix(trainingData_);

				// Collect all variable names
				const auto dependentVariableNames = getDependentVariableNames();
				const std::vector<std::string> indexedDependentVariableNames(dependentVariableNames.begin(), dependentVariableNames.end());

				// Train the models
				for(unsigned int dependentVariableIndex = 0; dependentVariableIndex < indexedDependentVariableNames.size(); ++dependentVariableIndex) {
					const auto& dependentVariableName = indexedDependentVariableNames[dependentVariableIndex];

					const auto& currentResponses = responses.row(dependentVariableIndex);

					// Train and store the model for the current dependent variable
					dependentVariableModels_[dependentVariableName] = mlpack::regression::LinearRegression(predictors, currentResponses);
				}
			}

			void LinearRegression::train(const std::map<std::string, double>& independentVariables, const std::map<std::string, double>& dependentVariables) {
				train({ { independentVariables, dependentVariables } });
			}

			std::vector<std::map<std::string, double>> LinearRegression::predict(const std::vector<std::map<std::string, double>>& data) const {
				// Collect all variable names
				const auto dependentVariableNames = getDependentVariableNames();

				// Generate the predictor variables matrix and response matrix
				const auto predictors = getMatrix(data);
				std::map<std::string, arma::rowvec> responses;
				for(const auto& dependentVariableName : dependentVariableNames) {
					// Predict the dependent variables
					responses[dependentVariableName] = arma::rowvec();
					dependentVariableModels_.at(dependentVariableName).Predict(predictors, responses[dependentVariableName]);
				}

				// Extract the dependent variables
				std::vector<std::map<std::string, double>> outputData = {};
				for(unsigned int row = 0; row < data.size(); ++row) {
					std::map<std::string, double> dependentVariables = {};

					for(const auto& dependentVariableName : dependentVariableNames) {
						dependentVariables[dependentVariableName] = responses[dependentVariableName][row];
					}

					outputData.push_back(dependentVariables);
				}

				return outputData;
			}

			std::map<std::string, double> LinearRegression::predict(const std::map<std::string, double>& independentVariables) const {
				return predict(std::vector<std::map<std::string, double>> { independentVariables })[0];
			}

			void LinearRegression::save(const std::string& filePath) {
				// Save training data
				std::ofstream ofstream(filePath);
				boost::archive::text_oarchive oarchive(ofstream);
				oarchive << trainingData_;

				// Save the models
				for(auto& dependentVariableModel : dependentVariableModels_) {
					auto& dependentVariable = dependentVariableModel.first;
					auto& model = dependentVariableModel.second;
					mlpack::data::Save(filePath + "-" + dependentVariable + ".txt", "lr_model", model);
				}
			}
		}
	}
}
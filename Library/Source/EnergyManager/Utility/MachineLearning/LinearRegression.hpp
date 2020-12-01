#pragma once

#include "EnergyManager/Utility/Logging/Loggable.hpp"

#include <map>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <set>
#include <string>
#include <vector>

namespace EnergyManager {
	namespace Utility {
		namespace MachineLearning {
			/**
			 * A statistical model that models the relation between a set of independent variables and a set of dependent variables.
			 */
			class LinearRegression : protected Utility::Logging::Loggable {
				/**
				 * Gets the corresponding matrix.
				 * @param data The data to use.
				 * @param defaultValue The default value to use for entries that miss certain variables.
				 * @return The corresponding matrix.
				 */
				static arma::mat getMatrix(const std::vector<std::map<std::string, double>>& data, const double& defaultValue = 0.0);

				/**
				 * Gets the independent variable matrix.
				 * @param data The data to use.
				 * @param defaultValue The default value to use for entries that miss certain variables.
				 * @return The independent variable matrix.
				 */
				static arma::mat getIndependentVariableMatrix(const std::map<std::map<std::string, double>, std::map<std::string, double>>& data);

				/**
				 * Gets the dependent variable matrix.
				 * @param data The data to use.
				 * @param defaultValue The default value to use for entries that miss certain variables.
				 * @return The dependent variable matrix.
				 */
				static arma::mat getDependentVariableMatrix(const std::map<std::map<std::string, double>, std::map<std::string, double>>& data);

				/**
				 * Gets the independent variables that are used by the data.
				 * @return The independent variables.
				 */
				static std::set<std::string> getVariableNames(const std::vector<std::map<std::string, double>>& data);

				/**
				 * Gets the independent variables that are used by the data.
				 * @return The independent variables.
				 */
				static std::set<std::string> getIndependentVariableNames(const std::map<std::map<std::string, double>, std::map<std::string, double>>& data);

				/**
				 * Gets the dependent variables that are used by the data.
				 * @return The dependent variables.
				 */
				static std::set<std::string> getDependentVariableNames(const std::map<std::map<std::string, double>, std::map<std::string, double>>& data);

				/**
				 * The training data used to train the model.
				 * Each entry of the map maps a set of independent variables to a set of dependent variables.
				 */
				std::map<std::map<std::string, double>, std::map<std::string, double>> trainingData_;

				/**
				 * The models for each dependent variable.
				 */
				std::map<std::string, mlpack::regression::LinearRegression> dependentVariableModels_;

			public:
				/**
				 * Creates a new LinearRegression.
				 */
				LinearRegression() = default;

				/**
				 * Creates a new LinearRegression.
				 * @param filePath The file to load the model data from.
				 * @param dependentVariableNames The dependent variable names.
				 */
				explicit LinearRegression(const std::string& filePath, const std::vector<std::string>& dependentVariableNames);

				/**
				 * Gets the training data.
				 * @return The training data.
				 */
				std::map<std::map<std::string, double>, std::map<std::string, double>> getTrainingData() const;

				/**
				 * Gets the independent variables that are used by the model.
				 * @return The independent variables.
				 */
				std::set<std::string> getIndependentVariableNames() const;

				/**
				 * Gets the dependent variables that are used by the model.
				 * @return The dependent variables.
				 */
				std::set<std::string> getDependentVariableNames() const;

				/**
				 * Resets the training data and clears the model.
				 */
				void reset();

				/**
				 * Re-trains the model using some new data.
				 * @param trainingData The new data.
				 */
				void train(const std::vector<std::pair<std::map<std::string, double>, std::map<std::string, double>>>& trainingData);

				/**
				 * Re-trains the model using some new data.
				 * @param independentVariables The independent variables of the training entry.
				 * @param dependentVariables The dependent variables of the training entry.
				 */
				void train(const std::map<std::string, double>& independentVariables, const std::map<std::string, double>& dependentVariables);

				/**
				 * Predicts the dependent variables given a set data
				 * @param data The data.
				 * @return The predicted dependent variables.
				 */
				std::vector<std::map<std::string, double>> predict(const std::vector<std::map<std::string, double>>& data) const;

				/**
				 * Predicts the dependent variables given a set of independent variables.
				 * @param independentVariables The independent variables.
				 * @return The predicted dependent variables.
				 */
				std::map<std::string, double> predict(const std::map<std::string, double>& independentVariables) const;

				/**
				 * Saves the model data.
				 * @param filePath The path to the file to save.
				 */
				void save(const std::string& filePath);
			};
		}
	}
}
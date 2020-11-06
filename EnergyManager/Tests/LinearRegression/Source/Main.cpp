#include <EnergyManager/Utility/Logging.hpp>
#include <EnergyManager/Utility/MachineLearning/LinearRegression.hpp>

int main(int argumentCount, char* argumentValues[]) {
	const int trainingOperations = 1000;

	// Create the model
	EnergyManager::Utility::MachineLearning::LinearRegression model;

	// Generate the training data
	std::map<std::map<std::string, double>, std::map<std::string, double>> trainingData = {};
	for(int i = 0; i < trainingOperations; ++i) {
		int number = std::rand() % 10000;

		// Teach square root
		int squareRoot = std::sqrt(number);

		trainingData[{ { "number", number } }] = { { "squareRoot", squareRoot } };
	}

	// Train the model
	model.train(trainingData);

	// Check predictions
	EnergyManager::Utility::Logging::logInformation("sqrt(25) = %f", model.predict({ { "number", 30 } })["squareRoot"]);
	//EnergyManager::Utility::Logging::logInformation("0 add 0 = %f", model.predict({ { "x", 0 }, { "y", 0 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("0 addW 0 = %f", model.predict({ { "x", 0 }, { "y", 0 } })["w"]);
	//EnergyManager::Utility::Logging::logInformation("0 add 5 = %f", model.predict({ { "x", 0 }, { "y", 5 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("0 addW 5 = %f", model.predict({ { "x", 0 }, { "y", 5 } })["w"]);
	//EnergyManager::Utility::Logging::logInformation("5 add 0 = %f", model.predict({ { "x", 5 }, { "y", 0 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("5 add 5 = %f", model.predict({ { "x", 5 }, { "y", 5 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("5 addW 5 = %f", model.predict({ { "x", 5 }, { "y", 5 } })["w"]);
	//EnergyManager::Utility::Logging::logInformation("10 add 0 = %f", model.predict({ { "x", 10 }, { "y", 0 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("0 add 10 = %f", model.predict({ { "x", 10 }, { "y", 0 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("10 add 10 = %f", model.predict({ { "x", 10 }, { "y", 10 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("10 add -10 = %f", model.predict({ { "x", 10 }, { "y", -10 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("15 add -10 = %f", model.predict({ { "x", 15 }, { "y", -10 } })["z"]);
	//EnergyManager::Utility::Logging::logInformation("15 addW -10 = %f", model.predict({ { "x", 15 }, { "y", -10 } })["w"]);

	return 0;
}
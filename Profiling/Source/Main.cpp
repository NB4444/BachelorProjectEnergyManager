#include "Jacobi.hpp"
#include "KMeans.hpp"
#include "MatrixMultiply.hpp"

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	// Run the tests
	//matrixMultiply(arguments);
	kMeans(arguments);
	//jacobi(arguments);

	return 0;
}
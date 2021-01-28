#include "BFS.hpp"
#include "Jacobi.hpp"
#include "KMeans.hpp"
#include "MatrixMultiply.hpp"

#include <EnergyManager/Hardware/Core.hpp>

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	// Run the tests
	bfs(arguments);
	jacobi(arguments);
	kMeans(arguments);
	matrixMultiply(arguments);

	return 0;
}
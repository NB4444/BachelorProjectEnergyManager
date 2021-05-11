#include "ExampleProfiler.hpp"

#include <EnergyManager.hpp>

int main(int argumentCount, char* argumentValues[]) {
	// We need to capture the command line arguments as some of those can be used to initialize the profiler
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	// Then we create a new instance of our profiler, to which we provide the command line arguments
	ExampleProfiler profiler(arguments);

	// Finally we can run the profiler
	profiler.run();

	return 0;
}

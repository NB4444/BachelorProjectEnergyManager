#include "ExampleProfiler.hpp"

#include <EnergyManager.hpp>

#include <chrono>

const auto energySavingInterval = std::chrono::milliseconds(10);

const auto halfingPeriod = 5 * energySavingInterval;

const auto doublingPeriod = 5 * energySavingInterval;

int main(int argumentCount, char* argumentValues[]) {
	// We need to capture the command line arguments as some of those can be used to initialize the profiler
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	// Then we create a new instance of our profiler, to which we provide the command line arguments
	ExampleProfiler profiler(arguments);

	profiler.addMonitor(std::make_shared<EnergyManager::Monitoring::Monitors::EnergyMonitor>(
		EnergyManager::Hardware::Core::getCore(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)),
		EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0)),
		energySavingInterval,
		true,
		halfingPeriod,
		doublingPeriod,
		true));

	// Finally we can run the profiler
	profiler.run();

	return 0;
}

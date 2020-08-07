#include "Application.hpp"
#include "Hardware/GPU.hpp"
#include "Testing/TestResults.hpp"
#include "Testing/TestRunner.hpp"
#include "Testing/Tests/PingTest.hpp"
#include "Utility/Logging.hpp"

#include <iostream>
#include <memory>
#include <unistd.h>

int main() {
	try {
		// Initialize APIs
		Hardware::GPU::initializeTracing();

		// Set up a new TestRunner
		Testing::TestRunner testRunner;

		// Start the executable
		bool running = true;
		auto monitor = std::thread([&] {
			Hardware::GPU gpu(0);

			while(running) {
				Utility::Logging::logInformation(
					"Monitored parameters:\n"
					"Fan speed: %d\n"
					"Memory clock: %d\n"
					"Power consumption: %d\n"
					"Power limit: %d\n"
					"Streaming multiprocessor clock: %d\n"
					"Temperature: %d",
					gpu.getFanSpeed(),
					gpu.getMemoryClock(),
					gpu.getPowerConsumption(),
					gpu.getPowerLimit(),
					gpu.getStreamingMultiprocessorClock(),
					gpu.getTemperature());

				sleep(1);
			}
		});

		// Add some tests
		testRunner.addTest(std::make_shared<Testing::Tests::PingTest>("google.com", 4));

		// Run the tests
		testRunner.run();

		running = false;
		monitor.join();
	} catch(const std::exception& exception) {
		Utility::Logging::logError(exception.what(), __FILE__, __LINE__);

		return 1;
	}

	return 0;
}
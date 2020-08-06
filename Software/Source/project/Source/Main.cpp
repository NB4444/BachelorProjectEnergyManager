#include "Application.hpp"
#include "Hardware/GPU.hpp"
#include "Testing/Test.hpp"
#include "Testing/TestResults.hpp"
#include "Testing/TestRunner.hpp"
#include "Utility/Logging.hpp"

#include <iostream>
#include <unistd.h>

int main() {
	try {
		// Set up a new TestRunner
		Testing::TestRunner testRunner;

		// Start the executable
		bool running = true;
		auto monitor = std::thread([&] {
			Hardware::GPU gpu;

			while(running) {
				Utility::Logging::logInformation(
					"Monitored parameters:\n"
					"Memory clock: %d\n"
					"Power consumption: %d\n"
					"Power limit: %d\n"
					"Streaming multiprocessor clock: %d\n"
					"Temperature: %d",
					gpu.getMemoryClock(),
					gpu.getPowerConsumption(),
					gpu.getPowerLimit(),
					gpu.getStreamingMultiprocessorClock(),
					gpu.getTemperature());

				sleep(1);
			}
		});

		// Add some tests
		testRunner.addTest(Testing::Test("Ping", Application("/bin/ping"), { "-c 4", "google.com" }, {
																										 { "Packets Transmitted", "(\\d+) packets transmitted" },
																										 { "Packets Received", "(\\d+) received" },
																										 { "Packets Lost", "(\\d+)% packet loss" },
																										 { "Time", "time (\\d+)" },
																									 }));

		// Run the tests
		testRunner.run();

		running = false;
		monitor.join();
	} catch(const std::exception& exception) {
		Utility::Logging::logError(exception.what(), __FILE__, __LINE__);

		return 1;
	} catch(const std::runtime_error& exception) {
		Utility::Logging::logError(exception.what(), __FILE__, __LINE__);

		return 1;
	}

	return 0;
}
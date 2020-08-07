#include "Application.hpp"
#include "Hardware/GPU.hpp"
#include "Testing/TestResults.hpp"
#include "Testing/TestRunner.hpp"
#include "Testing/Tests/PingTest.hpp"
#include "Testing/Tests/VectorAddSubtractTest.hpp"
#include "Utility/Logging.hpp"

#include <memory>
#include <unistd.h>

int main() {
	try {
		// Initialize APIs
		Hardware::GPU::initializeTracing();

		// Get the GPU
		auto gpu = Hardware::GPU::getGPU(0);

		// Set up a new TestRunner
		Testing::TestRunner testRunner;

		// Add some tests
		testRunner.addTest(std::make_shared<Testing::Tests::PingTest>("google.com", 10));
		testRunner.addTest(std::make_shared<Testing::Tests::VectorAddSubtractTest>());

		// Run the tests
		testRunner.run();

		return 0;
	} catch(const std::exception& exception) {
		Utility::Logging::logError(exception.what(), __FILE__, __LINE__);

		return 1;
	}
}
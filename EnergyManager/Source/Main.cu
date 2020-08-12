#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/TestResults.hpp"
#include "EnergyManager/Testing/TestRunner.hpp"
#include "EnergyManager/Testing/Tests/MatrixMultiplyTest.hpp"
#include "EnergyManager/Testing/Tests/PingTest.hpp"
#include "EnergyManager/Testing/Tests/VectorAddSubtractTest.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <memory>

int main() {
	try {
		// Initialize APIs
		EnergyManager::Hardware::GPU::initializeTracing();

		// Get the relevant hardware
		auto cpu = EnergyManager::Hardware::CPU::getCPU(0);
		auto gpu = EnergyManager::Hardware::GPU::getGPU(0);

		// Set up a new TestRunner
		EnergyManager::Testing::TestRunner testRunner;

		// Add some tests
		int multiplier = 50;
		//testRunner.addTest(std::make_shared<EnergyManager::Testing::Tests::PingTest>("google.com", 4));
		//testRunner.addTest(std::make_shared<EnergyManager::Testing::Tests::VectorAddSubtractTest>(*gpu, 50000));
		testRunner.addTest(std::make_shared<EnergyManager::Testing::Tests::MatrixMultiplyTest>(*gpu, 32 * multiplier, 32 * multiplier, 32 * multiplier, 32 * multiplier));

		// Run the tests
		testRunner.run();

		return 0;
	} catch(const std::exception& exception) {
		EnergyManager::Utility::Logging::logError(exception.what(), __FILE__, __LINE__);

		return 1;
	}
}
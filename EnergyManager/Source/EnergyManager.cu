#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/TestResults.hpp"
#include "EnergyManager/Testing/TestRunner.hpp"
#include "EnergyManager/Testing/Tests/FixedFrequencyMatrixMultiplyTest.hpp"
#include "EnergyManager/Testing/Tests/MatrixMultiplyTest.hpp"
#include "EnergyManager/Testing/Tests/PingTest.hpp"
#include "EnergyManager/Testing/Tests/VectorAddSubtractTest.hpp"
#include "EnergyManager/Utility/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"

#include <getopt.h>
#include <iostream>
#include <memory>

std::map<std::string, std::string> parseArguments(int argumentCount, char* argumentValues[]) {
	std::map<std::string, std::string> arguments;

	auto showUsage = [&](auto& stream) {
		stream << "usage: " << argumentValues[0] << " [-h] --database FILE" << std::endl
			   << std::endl
			   << "Manage GPU energy while running various workloads." << std::endl
			   << std::endl
			   << "optional arguments:" << std::endl
			   << std::endl
			   << "\t-h, --help\t\t\t\t\tshow this help message and exit" << std::endl
			   << std::endl
			   << std::endl
			   << "output:" << std::endl
			   << std::endl
			   << "\t--database FILE, -d FILE\tSpecifies the database to use." << std::endl;
	};

	size_t argumentIndex = 0u;
	while(argumentIndex < argumentCount) {
		std::string value = argumentValues[argumentIndex];

		if(value == "--help" || value == "-h") {
			showUsage(std::cout);

			exit(0);
		} else if(value == "--database" || value == "-d") {
			arguments["database"] = argumentValues[++argumentIndex];
		}

		++argumentIndex;
	}

	if(arguments.find("database") == arguments.end()) {
		showUsage(std::cerr);

		exit(1);
	}

	return arguments;
}

int main(int argumentCount, char* argumentValues[]) {
	// Parse the arguments
	std::map<std::string, std::string> arguments = parseArguments(argumentCount, argumentValues);

	try {
		// Initialize APIs
		EnergyManager::Hardware::GPU::initializeTracing();
		EnergyManager::Persistence::Entity::setDatabaseFile(arguments["database"]);

		// Get the relevant hardware
		auto cpu = EnergyManager::Hardware::CPU::getCPU(0);
		auto gpu = EnergyManager::Hardware::GPU::getGPU(0);

		// Set up a new TestRunner
		EnergyManager::Testing::TestRunner testRunner;

		// Add some tests
		//testRunner.addTest(std::make_shared<EnergyManager::Testing::Tests::PingTest>("google.com", 4));
		//testRunner.addTest(std::make_shared<EnergyManager::Testing::Tests::VectorAddSubtractTest>(*gpu, 50000));
		//testRunner.addTest(std::make_shared<EnergyManager::Testing::Tests::MatrixMultiplyTest>(*gpu, 32 * multiplier, 32 * multiplier, 32 * multiplier, 32 * multiplier));

		//EnergyManager::Utility::Logging::logInformation("GPU frequency: %d", gpu->getCoreClockRate());

		const int sizeMultiplier = 50;
		const size_t testSegments = 4;

		double cpuClockRatePerSegment = static_cast<double>(cpu->getMaximumCoreClockRate()) / testSegments;
		double gpuClockRatePerSegment = static_cast<double>(gpu->getMaximumCoreClockRate()) / testSegments;

		for(size_t segmentIndex = 0u; segmentIndex < testSegments; ++segmentIndex) {
			long cpuLowerFrequency = segmentIndex * cpuClockRatePerSegment;
			long cpuUpperFrequency = (segmentIndex + 1) * cpuClockRatePerSegment;
			long gpuLowerFrequency = segmentIndex * gpuClockRatePerSegment;
			long gpuUpperFrequency = (segmentIndex + 1) * gpuClockRatePerSegment;

			testRunner.addTest(std::make_shared<EnergyManager::Testing::Tests::FixedFrequencyMatrixMultiplyTest>(
				"Fixed Frequency Matrix Multiply (CPU: " + std::to_string(cpuLowerFrequency) + "-" + std::to_string(cpuUpperFrequency) + " kHz | GPU: " + std::to_string(gpuLowerFrequency) + "-"
					+ std::to_string(gpuUpperFrequency) + " kHz)",
				*cpu,
				*gpu,
				32 * sizeMultiplier,
				32 * sizeMultiplier,
				32 * sizeMultiplier,
				32 * sizeMultiplier,
				cpuLowerFrequency,
				cpuUpperFrequency,
				gpuLowerFrequency,
				gpuUpperFrequency));
		}

		testRunner.addTest(
			std::make_shared<EnergyManager::Testing::Tests::MatrixMultiplyTest>("Matrix Multiply", *cpu, *gpu, 32 * sizeMultiplier, 32 * sizeMultiplier, 32 * sizeMultiplier, 32 * sizeMultiplier));

		// Run the tests
		testRunner.run(arguments["database"]);

		// TODO: Pretty-print the test results to the console

		return 0;
	} catch(const EnergyManager::Utility::Exception& exception) {
		EnergyManager::Utility::Logging::logError(exception.getMessage(), exception.getFile(), exception.getLine());

		return 1;
	} catch(const std::exception& exception) {
		EnergyManager::Utility::Logging::logError(exception.what(), __FILE__, __LINE__);

		return 1;
	}
}
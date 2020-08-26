#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Testing/TestResults.hpp"
#include "EnergyManager/Testing/TestRunner.hpp"
#include "EnergyManager/Testing/Tests/FixedFrequencyMatrixMultiplyTest.hpp"
#include "EnergyManager/Testing/Tests/MatrixMultiplyTest.hpp"
#include "EnergyManager/Testing/Tests/PingTest.hpp"
#include "EnergyManager/Testing/Tests/VectorAddSubtractTest.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <getopt.h>
#include <iostream>
#include <memory>

auto parseArguments(int argumentCount, char* argumentValues[]) {
	auto showUsage = [&](auto& stream) {
		stream << "Usage: " << argumentValues[0] << " [-h] --database FILE" << std::endl
			<< std::endl
			<< "Manage GPU energy while running various workloads." << std::endl
			<< std::endl
			<< "Optional Arguments:" << std::endl
			<< std::endl
			<< "\t-h, --help\t\t\t\t\tShow this help message and exit" << std::endl
			<< std::endl
			<< std::endl
			<< "Testing:" << std::endl
			<< std::endl
			<< "\t--test NAME, -t NAME\tSpecifies the test to run." << std::endl
			<< "\t--parameter NAME=VALUE, -p NAME=VALUE\tSpecifies a parameter and its associated value to provide to the test." << std::endl
			<< std::endl
			<< std::endl
			<< "Output:" << std::endl
			<< std::endl
			<< "\t--database FILE, -d FILE\tSpecifies the database to use." << std::endl;
	};

	struct {
		std::string database = "database.sqlite";
		std::string test = "";
		std::map<std::string, std::string> parameters = {};
	} arguments;
	size_t argumentIndex = 0u;
	while(argumentIndex < argumentCount) {
		std::string value = argumentValues[argumentIndex];

		if(value == "--help" || value == "-h") {
			showUsage(std::cout);

			exit(0);
		} else if(value == "--database" || value == "-d") {
			arguments.database = argumentValues[++argumentIndex];
		} else if(value == "--test" || value == "-t") {
			arguments.test = argumentValues[++argumentIndex];
		} else if(value == "--parameter" || value == "-p") {
			std::vector<std::string> parameter = EnergyManager::Utility::Text::splitToVector(argumentValues[++argumentIndex], "=", true);

			arguments.parameters[parameter[0]] = parameter[1];
		}

		++argumentIndex;
	}

	return arguments;
}

int main(int argumentCount, char* argumentValues[]) {
	// Parse the arguments
	auto arguments = parseArguments(argumentCount, argumentValues);

	try {
		// Initialize APIs
		EnergyManager::Utility::Exceptions::Exception::initialize();
		EnergyManager::Hardware::GPU::initialize();
		EnergyManager::Persistence::Entity::initialize(arguments.database);

		// Set up a new TestRunner
		EnergyManager::Testing::TestRunner testRunner;

		// Generate the test
		testRunner.addTest(std::shared_ptr<EnergyManager::Testing::Tests::Test>([&]() -> EnergyManager::Testing::Tests::Test* {
			if(arguments.test == "FixedFrequencyMatrixMultiplyTest") {
				return new EnergyManager::Testing::Tests::FixedFrequencyMatrixMultiplyTest(
					arguments.parameters["name"],
					EnergyManager::Hardware::CPU::getCPU(std::stoi(arguments.parameters["cpu"])),
					EnergyManager::Hardware::GPU::getGPU(std::stoi(arguments.parameters["gpu"])),
					std::stoi(arguments.parameters["matrixAWidth"]),
					std::stoi(arguments.parameters["matrixAHeight"]),
					std::stoi(arguments.parameters["matrixBWidth"]),
					std::stoi(arguments.parameters["matrixBHeight"]),
					std::stoul(arguments.parameters["minimumCPUFrequency"]),
					std::stoul(arguments.parameters["maximumCPUFrequency"]),
					std::stoul(arguments.parameters["minimumGPUFrequency"]),
					std::stoul(arguments.parameters["maximumGPUFrequency"]));
			} else if(arguments.test == "MatrixMultiplyTest") {
				return new EnergyManager::Testing::Tests::MatrixMultiplyTest(
					arguments.parameters["name"],
					EnergyManager::Hardware::CPU::getCPU(std::stoi(arguments.parameters["cpu"])),
					EnergyManager::Hardware::GPU::getGPU(std::stoi(arguments.parameters["gpu"])),
					std::stoi(arguments.parameters["matrixAWidth"]),
					std::stoi(arguments.parameters["matrixAHeight"]),
					std::stoi(arguments.parameters["matrixBWidth"]),
					std::stoi(arguments.parameters["matrixBHeight"]));
			} else if(arguments.test == "PingTest") {
				return new EnergyManager::Testing::Tests::PingTest(arguments.parameters["name"], arguments.parameters["host"], std::stoi(arguments.parameters["times"]));
			} else if(arguments.test == "VectorAddSubtractTest") {
				return new EnergyManager::Testing::Tests::VectorAddSubtractTest(
					arguments.parameters["name"],
					EnergyManager::Hardware::GPU::getGPU(std::stoi(arguments.parameters["gpu"])),
					std::stoi(arguments.parameters["computeCount"]));
			}

			ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Unknown test type");
		}()));

		// Run the tests
		testRunner.run(arguments.database);

		return 0;
	} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
		exception.log();

		return 1;
	} catch(const std::exception& exception) {
		EnergyManager::Utility::Exceptions::Exception(exception.what(), __FILE__, __LINE__).log();

		return 1;
	}
}
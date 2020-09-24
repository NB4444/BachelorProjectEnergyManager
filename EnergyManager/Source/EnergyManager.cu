#include "EnergyManager/Benchmarking/Workloads/ActiveInactiveWorkload.hpp"
#include "EnergyManager/Benchmarking/Workloads/AllocateFreeWorkload.hpp"
#include "EnergyManager/Benchmarking/Workloads/SyntheticWorkload.hpp"
#include "EnergyManager/Benchmarking/Workloads/VectorAddWorkload.hpp"
#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Hardware/Node.hpp"
#include "EnergyManager/Monitoring/ApplicationMonitor.hpp"
#include "EnergyManager/Monitoring/CPUMonitor.hpp"
#include "EnergyManager/Monitoring/GPUMonitor.hpp"
#include "EnergyManager/Monitoring/NodeMonitor.hpp"
#include "EnergyManager/Testing/TestResults.hpp"
#include "EnergyManager/Testing/TestRunner.hpp"
#include "EnergyManager/Testing/Tests/FixedFrequencyMatrixMultiplyTest.hpp"
#include "EnergyManager/Testing/Tests/MatrixMultiplyTest.hpp"
#include "EnergyManager/Testing/Tests/PingTest.hpp"
#include "EnergyManager/Testing/Tests/SyntheticWorkloadTest.hpp"
#include "EnergyManager/Testing/Tests/VectorAddSubtractTest.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <iostream>
#include <memory>

struct MonitorConfiguration {
	std::string monitor = "";
	std::map<std::string, std::string> parameters = {};
};

struct TestConfiguration {
	std::string test = "";
	std::map<std::string, std::string> parameters = {};
	std::vector<MonitorConfiguration> monitorConfigurations = {};
};

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
			   << "\t--monitor NAME, -m NAME\tSpecifies a monitor to use." << std::endl
			   << "\t--monitorParameter NAME=VALUE, -mP NAME=VALUE\tSpecifies a parameter and its associated value to provide to the monitor." << std::endl
			   << std::endl
			   << std::endl
			   << "Output:" << std::endl
			   << std::endl
			   << "\t--database FILE, -d FILE\tSpecifies the database to use." << std::endl;
	};

	// Keep track of arguments
	struct {
		std::string database = "database.sqlite";
		std::vector<TestConfiguration> testConfigurations = {};
	} arguments;
	TestConfiguration* currentTestConfiguration = nullptr;
	MonitorConfiguration* currentMonitorConfiguration = nullptr;

	// Parse arguments
	size_t argumentIndex = 0u;
	while(argumentIndex < argumentCount) {
		std::string value = argumentValues[argumentIndex];

		if(value == "--help" || value == "-h") {
			showUsage(std::cout);

			exit(0);
		} else if(value == "--database" || value == "-d") {
			arguments.database = argumentValues[++argumentIndex];
		} else if(value == "--test" || value == "-t") {
			arguments.testConfigurations.emplace_back(TestConfiguration { .test = argumentValues[++argumentIndex] });
			currentTestConfiguration = &arguments.testConfigurations[arguments.testConfigurations.size() - 1];
		} else if((value == "--parameter" || value == "-p") && currentTestConfiguration != nullptr) {
			std::vector<std::string> parameter = EnergyManager::Utility::Text::splitToVector(argumentValues[++argumentIndex], "=", true);

			currentTestConfiguration->parameters[parameter[0]] = parameter[1];
		} else if(value == "--monitor" || value == "-m") {
			currentTestConfiguration->monitorConfigurations.emplace_back(MonitorConfiguration { .monitor = argumentValues[++argumentIndex] });
			currentMonitorConfiguration = &currentTestConfiguration->monitorConfigurations[currentTestConfiguration->monitorConfigurations.size() - 1];
		} else if((value == "--monitorParameter" || value == "-mP") && currentMonitorConfiguration != nullptr) {
			std::vector<std::string> parameter = EnergyManager::Utility::Text::splitToVector(argumentValues[++argumentIndex], "=", true);

			currentMonitorConfiguration->parameters[parameter[0]] = parameter[1];
		}

		++argumentIndex;
	}

	return arguments;
}

std::shared_ptr<EnergyManager::Testing::Tests::Test> parseTest(TestConfiguration& testConfiguration) {
	// Parse the monitors
	std::map<std::shared_ptr<EnergyManager::Monitoring::Monitor>, std::chrono::system_clock::duration> monitors;
	for(auto& monitorConfiguration : testConfiguration.monitorConfigurations) {
		monitors.insert({
			EnergyManager::Monitoring::Monitor::parse(monitorConfiguration.monitor, monitorConfiguration.parameters),
			std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(std::stoul(monitorConfiguration.parameters["pollingInterval"]))),
		});
	}

	// Parse the test
	return EnergyManager::Testing::Tests::Test::parse(testConfiguration.test, testConfiguration.parameters, monitors);
}

int main(int argumentCount, char* argumentValues[]) {
	// Parse the arguments
	auto arguments = parseArguments(argumentCount, argumentValues);

	try {
		// Initialize utility framework
		EnergyManager::Utility::Exceptions::Exception::initialize();

		// Initialize hardware framework
		EnergyManager::Hardware::GPU::initialize();

		// Initialize benchmarking framework
		EnergyManager::Benchmarking::Workloads::ActiveInactiveWorkload::initialize();
		EnergyManager::Benchmarking::Workloads::AllocateFreeWorkload::initialize();
		EnergyManager::Benchmarking::Workloads::VectorAddWorkload::initialize();

		// Initialize monitoring framework
		EnergyManager::Monitoring::CPUMonitor::initialize();
		EnergyManager::Monitoring::GPUMonitor::initialize();
		EnergyManager::Monitoring::NodeMonitor::initialize();

		// Initialize persistence framework
		EnergyManager::Persistence::Entity::initialize(arguments.database);

		// Initialize testing framework
		EnergyManager::Testing::Tests::FixedFrequencyMatrixMultiplyTest::initialize();
		EnergyManager::Testing::Tests::MatrixMultiplyTest::initialize();
		EnergyManager::Testing::Tests::PingTest::initialize();
		EnergyManager::Testing::Tests::SyntheticWorkloadTest::initialize();
		EnergyManager::Testing::Tests::VectorAddSubtractTest::initialize();

		// Set up a new TestRunner
		EnergyManager::Testing::TestRunner testRunner;

		// Generate the tests
		for(auto& testConfiguration : arguments.testConfigurations) {
			testRunner.addTest(parseTest(testConfiguration));
		}

		// Run the tests
		testRunner.run();

		return 0;
	} catch(const EnergyManager::Utility::Exceptions::Exception& exception) {
		exception.log();

		return 1;
	} catch(const std::exception& exception) {
		EnergyManager::Utility::Exceptions::Exception(exception.what(), __FILE__, __LINE__).log();

		return 1;
	}
}
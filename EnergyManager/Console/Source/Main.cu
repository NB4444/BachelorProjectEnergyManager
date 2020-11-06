#include <EnergyManager/Benchmarking/Workloads/ActiveInactiveWorkload.hpp>
#include <EnergyManager/Benchmarking/Workloads/AllocateFreeWorkload.hpp>
#include <EnergyManager/Benchmarking/Workloads/VectorAddWorkload.hpp>
#include <EnergyManager/Benchmarking/Workloads/Workload.hpp>
#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Hardware/Node.hpp>
#include <EnergyManager/Monitoring/Monitors/ApplicationMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/CPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/GPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/NodeMonitor.hpp>
#include <EnergyManager/Testing/Persistence/TestResults.hpp>
#include <EnergyManager/Testing/TestRunner.hpp>
#include <EnergyManager/Testing/Tests/FixedFrequencyMatrixMultiplyTest.hpp>
#include <EnergyManager/Testing/Tests/MatrixMultiplyTest.hpp>
#include <EnergyManager/Testing/Tests/PingTest.hpp>
#include <EnergyManager/Testing/Tests/VectorAddSubtractTest.hpp>
#include <EnergyManager/Testing/Tests/WorkloadTest.hpp>
#include <EnergyManager/Utility/Exceptions/Exception.hpp>
#include <EnergyManager/Utility/Text.hpp>
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
			   << "\t--monitor NAME, -m NAME\tSpecifies a monitor to use." << std::endl
			   << "\t--parameter NAME=VALUE, -p NAME=VALUE\tSpecifies a parameter and its associated value. Used to instantiate the last provided test or monitor." << std::endl
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
			currentMonitorConfiguration = nullptr;
		} else if((value == "--parameter" || value == "-p") && currentTestConfiguration != nullptr) {
			if(currentMonitorConfiguration != nullptr) {
				std::vector<std::string> parameter = EnergyManager::Utility::Text::splitToVector(argumentValues[++argumentIndex], "=", true);

				currentMonitorConfiguration->parameters[parameter[0]] = parameter[1];
			} else {
				std::vector<std::string> parameter = EnergyManager::Utility::Text::splitToVector(argumentValues[++argumentIndex], "=", true);

				currentTestConfiguration->parameters[parameter[0]] = parameter[1];
			}
		} else if(value == "--monitor" || value == "-m") {
			currentTestConfiguration->monitorConfigurations.emplace_back(MonitorConfiguration { .monitor = argumentValues[++argumentIndex] });
			currentMonitorConfiguration = &currentTestConfiguration->monitorConfigurations[currentTestConfiguration->monitorConfigurations.size() - 1];
		}

		++argumentIndex;
	}

	return arguments;
}

std::shared_ptr<EnergyManager::Testing::Tests::Test> parseTest(const TestConfiguration& testConfiguration) {
	// Parse the monitors
	std::map<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>, std::chrono::system_clock::duration> monitors;
	for(auto& monitorConfiguration : testConfiguration.monitorConfigurations) {
		monitors.insert({
			EnergyManager::Monitoring::Monitor::parse(monitorConfiguration.monitor, monitorConfiguration.parameters),
			std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(std::stoul(monitorConfiguration.parameters.at("pollingInterval")))),
		});
	}

	// Parse the test
	return EnergyManager::Testing::Tests::Test::parse(testConfiguration.test, testConfiguration.parameters, monitors);
}

int main(int argumentCount, char* argumentValues[]) {
	// Parse the arguments
	auto arguments = parseArguments(argumentCount, argumentValues);

	// Load the database
	EnergyManager::Persistence::Entity::initialize(arguments.database);

	// Parse the tests
	std::vector<std::shared_ptr<EnergyManager::Testing::Tests::Test>> tests;
	std::transform(arguments.testConfigurations.begin(), arguments.testConfigurations.end(), std::back_inserter(tests), [](const auto& testConfiguration) {
		return parseTest(testConfiguration);
	});

	// Set up a new TestRunner
	EnergyManager::Testing::TestRunner testRunner;
	for(const auto& test : tests) {
		testRunner.addTest(test);
	}

	// Run the tests
	testRunner.run();

	return 0;
}
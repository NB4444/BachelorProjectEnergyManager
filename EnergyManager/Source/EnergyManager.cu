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
#include "EnergyManager/Testing/Tests/SyntheticGPUWorkloadTest.hpp"
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

std::pair<std::shared_ptr<EnergyManager::Monitoring::Monitor>, std::chrono::system_clock::duration> parseMonitor(MonitorConfiguration& monitorConfiguration) {
	std::shared_ptr<EnergyManager::Monitoring::Monitor> monitor;
	if(monitorConfiguration.monitor == "CPUMonitor") {
		monitor = std::make_shared<EnergyManager::Monitoring::CPUMonitor>(EnergyManager::Hardware::CPU::getCPU(std::stoi(monitorConfiguration.parameters["cpu"])));
	} else if(monitorConfiguration.monitor == "GPUMonitor") {
		monitor = std::make_shared<EnergyManager::Monitoring::GPUMonitor>(EnergyManager::Hardware::GPU::getGPU(std::stoi(monitorConfiguration.parameters["gpu"])));
	} else if(monitorConfiguration.monitor == "NodeMonitor") {
		monitor = std::make_shared<EnergyManager::Monitoring::NodeMonitor>(EnergyManager::Hardware::Node::getNode());
	}

	return {
		monitor,
		std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(std::stoul(monitorConfiguration.parameters["pollingInterval"]))),
	};
}

std::shared_ptr<EnergyManager::Testing::Tests::Test> parseTest(TestConfiguration& testConfiguration) {
	// Parse the monitors
	std::map<std::shared_ptr<EnergyManager::Monitoring::Monitor>, std::chrono::system_clock::duration> monitors;
	for(auto& monitorConfiguration : testConfiguration.monitorConfigurations) {
		monitors.insert(parseMonitor(monitorConfiguration));
	}

	// Parse the tests
	if(testConfiguration.test == "FixedFrequencyMatrixMultiplyTest") {
		return std::make_shared<EnergyManager::Testing::Tests::FixedFrequencyMatrixMultiplyTest>(
			testConfiguration.parameters["name"],
			EnergyManager::Hardware::Node::getNode(),
			EnergyManager::Hardware::CPU::getCPU(std::stoi(testConfiguration.parameters["cpu"])),
			EnergyManager::Hardware::GPU::getGPU(std::stoi(testConfiguration.parameters["gpu"])),
			std::stoi(testConfiguration.parameters["matrixAWidth"]),
			std::stoi(testConfiguration.parameters["matrixAHeight"]),
			std::stoi(testConfiguration.parameters["matrixBWidth"]),
			std::stoi(testConfiguration.parameters["matrixBHeight"]),
			std::stoul(testConfiguration.parameters["minimumCPUFrequency"]),
			std::stoul(testConfiguration.parameters["maximumCPUFrequency"]),
			std::stoul(testConfiguration.parameters["minimumGPUFrequency"]),
			std::stoul(testConfiguration.parameters["maximumGPUFrequency"]),
			std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(std::stoul(testConfiguration.parameters["applicationMonitorPollingInterval"]))),
			monitors);
	} else if(testConfiguration.test == "MatrixMultiplyTest") {
		return std::make_shared<EnergyManager::Testing::Tests::MatrixMultiplyTest>(
			testConfiguration.parameters["name"],
			EnergyManager::Hardware::Node::getNode(),
			EnergyManager::Hardware::CPU::getCPU(std::stoi(testConfiguration.parameters["cpu"])),
			EnergyManager::Hardware::GPU::getGPU(std::stoi(testConfiguration.parameters["gpu"])),
			std::stoi(testConfiguration.parameters["matrixAWidth"]),
			std::stoi(testConfiguration.parameters["matrixAHeight"]),
			std::stoi(testConfiguration.parameters["matrixBWidth"]),
			std::stoi(testConfiguration.parameters["matrixBHeight"]),
			std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(std::stoul(testConfiguration.parameters["applicationMonitorPollingInterval"]))),
			monitors);
	} else if(testConfiguration.test == "PingTest") {
		return std::make_shared<EnergyManager::Testing::Tests::PingTest>(
			testConfiguration.parameters["name"],
			testConfiguration.parameters["host"],
			std::stoi(testConfiguration.parameters["times"]),
			std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(std::stoul(testConfiguration.parameters["applicationMonitorPollingInterval"]))),
			monitors);
	} else if(testConfiguration.test == "SyntheticGPUWorkloadTest") {
		auto workloadName = testConfiguration.parameters["workload"];
		std::shared_ptr<EnergyManager::Benchmarking::Workloads::SyntheticGPUWorkload> workload;
		if(workloadName == "ActiveInactiveWorkload") {
			workload = std::make_shared<EnergyManager::Benchmarking::Workloads::ActiveInactiveWorkload>(
				std::stoi(testConfiguration.parameters["activeOperations"]),
				std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::milliseconds(std::stol(testConfiguration.parameters["inactivePeriod"]))),
				std::stoi(testConfiguration.parameters["cycles"]));
		} else if(workloadName == "AllocateFreeWorkload") {
			workload = std::make_shared<EnergyManager::Benchmarking::Workloads::AllocateFreeWorkload>(
				std::stoi(testConfiguration.parameters["hostAllocations"]),
				std::stol(testConfiguration.parameters["hostSize"]),
				std::stoi(testConfiguration.parameters["deviceAllocations"]),
				std::stol(testConfiguration.parameters["deviceSize"]));
		} else if(workloadName == "VectorAddWorkload") {
			workload = std::make_shared<EnergyManager::Benchmarking::Workloads::VectorAddWorkload>(std::stol(testConfiguration.parameters["size"]));
		}

		return std::make_shared<EnergyManager::Testing::Tests::SyntheticGPUWorkloadTest>(
			testConfiguration.parameters["name"],
			workload,
			EnergyManager::Hardware::Node::getNode(),
			EnergyManager::Hardware::CPU::getCPU(std::stoi(testConfiguration.parameters["cpu"])),
			EnergyManager::Hardware::GPU::getGPU(std::stoi(testConfiguration.parameters["gpu"])),
			monitors);
	} else if(testConfiguration.test == "VectorAddSubtractTest") {
		return std::make_shared<EnergyManager::Testing::Tests::VectorAddSubtractTest>(
			testConfiguration.parameters["name"],
			EnergyManager::Hardware::GPU::getGPU(std::stoi(testConfiguration.parameters["gpu"])),
			std::stoi(testConfiguration.parameters["computeCount"]),
			monitors);
	}

	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Unknown test type");
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
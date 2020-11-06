#include "Workloads/ActiveInactiveWorkload.hpp"

#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Hardware/Node.hpp>
#include <EnergyManager/Monitoring/Monitors/CPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/GPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/NodeMonitor.hpp>
#include <EnergyManager/Testing/TestRunner.hpp>
#include <EnergyManager/Testing/Tests/WorkloadTest.hpp>
#include <EnergyManager/Utility/Text.hpp>
#include <memory>

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);
	const auto database = EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--database", std::string(PROJECT_RESOURCES_DIRECTORY) + "/Test Results/database.sqlite");
	const auto name = EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--name", "Active Inactive Workload Test");
	const auto activeOperations = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--activeOperations", 5000);
	const auto inactivePeriod = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--inactivePeriod", std::chrono::milliseconds(2500));
	const auto cycles = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--cycles", 5);

	const auto monitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(500));
	const auto nodeMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--nodeMonitorInterval", monitorInterval);
	const auto cpuMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--cpuMonitorInterval", monitorInterval);
	const auto gpuMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--gpuMonitorInterval", monitorInterval);

	// Add monitors
	std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>> monitors
		= { std::make_shared<EnergyManager::Monitoring::NodeMonitor>(EnergyManager::Hardware::Node::getNode(), nodeMonitorInterval) };
	for(const auto& cpu : EnergyManager::Hardware::CPU::getCPUs()) {
		monitors.push_back(std::make_shared<EnergyManager::Monitoring::CPUMonitor>(cpu, cpuMonitorInterval));
	}
	for(const auto& gpu : EnergyManager::Hardware::GPU::getGPUs()) {
		monitors.push_back(std::make_shared<EnergyManager::Monitoring::GPUMonitor>(gpu, gpuMonitorInterval));
	}

	// Load the database
	EnergyManager::Persistence::Entity::initialize(database);

	// Set up a new TestRunner
	EnergyManager::Testing::TestRunner testRunner(
		{ std::make_shared<EnergyManager::Testing::Tests::WorkloadTest>(name, std::make_shared<Workloads::ActiveInactiveWorkload>(activeOperations, inactivePeriod, cycles), monitors) });

	// Run the tests
	testRunner.run();

	return 0;
}
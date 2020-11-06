#include "Tests/FixedFrequencyMatrixMultiplyTest.hpp"

#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Hardware/Node.hpp>
#include <EnergyManager/Monitoring/Monitors/CPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/GPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/NodeMonitor.hpp>
#include <EnergyManager/Testing/TestRunner.hpp>
#include <memory>

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);
	const auto database = EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--database", std::string(PROJECT_RESOURCES_DIRECTORY) + "/Test Results/database.sqlite");
	const auto name = EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--name", "Fixed Frequency Matrix Multiply Test");
	const auto cpus = EnergyManager::Hardware::CPU::parseCPUs(EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--cpus", "0"));
	const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));
	const auto sizeMultiplier = EnergyManager::Utility::Text::getArgument(arguments, "--sizeMultiplier", 100);
	const auto matrixAWidth = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixAWidth", 32 * sizeMultiplier);
	const auto matrixAHeight = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixAHeight", 32 * sizeMultiplier);
	const auto matrixBWidth = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixBWidth", 32 * sizeMultiplier);
	const auto matrixBHeight = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixBHeight", 32 * sizeMultiplier);
	const auto minimumCPUFrequency = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--minimumCPUFrequency", 0);
	const auto maximumCPUFrequency = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--maximumCPUFrequency", -1);
	const auto minimumGPUFrequency = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--minimumGPUFrequency", 0);
	const auto maximumGPUFrequency = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--maximumGPUFrequency", -1);

	const auto monitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(500));
	const auto applicationMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--applicationMonitorInterval", monitorInterval);
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
	EnergyManager::Testing::TestRunner testRunner({ std::make_shared<Tests::FixedFrequencyMatrixMultiplyTest>(
		name,
		cpus,
		gpu,
		matrixAHeight,
		matrixAWidth,
		matrixBWidth,
		matrixBHeight,
		minimumCPUFrequency,
		maximumCPUFrequency,
		minimumGPUFrequency,
		maximumGPUFrequency,
		applicationMonitorInterval,
		monitors) });

	// Run the tests
	testRunner.run();

	return 0;
}
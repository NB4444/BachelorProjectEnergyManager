#include <EnergyManager/EnergySaving/EnergyManager.hpp>
#include <EnergyManager/EnergySaving/Models/StaticDVFSModel.hpp>
#include <EnergyManager/EnergySaving/Strategies/DVFSStrategy.hpp>
#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Hardware/Node.hpp>
#include <EnergyManager/Monitoring/Monitors/CPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/GPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/NodeMonitor.hpp>
#include <EnergyManager/Testing/TestRunner.hpp>
#include <EnergyManager/Testing/Tests/ApplicationTest.hpp>
#include <memory>

std::shared_ptr<EnergyManager::EnergySaving::Models::DVFSModel> train(
	EnergyManager::Utility::Application& application,
	const std::chrono::system_clock::duration& cpuTrainingMonitorInterval,
	const std::chrono::system_clock::duration& gpuTrainingMonitorInterval,
	const std::string& dvfsModelPath) {
	// Set up monitors to use for training
	auto cpu = EnergyManager::Hardware::CPU::getCPU(0);
	auto gpu = EnergyManager::Hardware::GPU::getGPU(0);
	auto cpuMonitor = std::make_shared<EnergyManager::Monitoring::CPUMonitor>(cpu, cpuTrainingMonitorInterval);
	auto gpuMonitor = std::make_shared<EnergyManager::Monitoring::GPUMonitor>(gpu, gpuTrainingMonitorInterval);

	// Set up a model to use for the energy manager's algorithm
	const auto dvfsModel = std::make_shared<EnergyManager::EnergySaving::Models::StaticDVFSModel>(cpu, gpu, cpuMonitor, gpuMonitor, dvfsModelPath, [&] {
		application.run();
	});

	// Next, profile the application to train the model
	//dvfsModel->run();

	return dvfsModel;
}

void test(
	const std::string& name,
	EnergyManager::Utility::Application& application,
	const std::chrono::system_clock::duration& applicationMonitorInterval,
	const std::chrono::system_clock::duration& nodeMonitorInterval,
	const std::chrono::system_clock::duration& cpuMonitorInterval,
	const std::chrono::system_clock::duration& gpuMonitorInterval,
	const std::shared_ptr<EnergyManager::EnergySaving::Models::DVFSModel>& dvfsModel,
	const std::chrono::system_clock::duration& dvfsStrategyInterval,
	const std::vector<std::shared_ptr<EnergyManager::Hardware::CPU>>& cpus,
	const std::shared_ptr<EnergyManager::Hardware::GPU>& gpu) {
	// Add monitors
	std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>> monitors
		= { std::make_shared<EnergyManager::Monitoring::NodeMonitor>(EnergyManager::Hardware::Node::getNode(), nodeMonitorInterval) };
	std::vector<std::shared_ptr<EnergyManager::Monitoring::CPUMonitor>> cpuMonitors = {};
	for(const auto& cpu : EnergyManager::Hardware::CPU::getCPUs()) {
		auto monitor = std::make_shared<EnergyManager::Monitoring::CPUMonitor>(cpu, cpuMonitorInterval);
		monitors.push_back(monitor);
		cpuMonitors.push_back(monitor);
	}
	std::vector<std::shared_ptr<EnergyManager::Monitoring::GPUMonitor>> gpuMonitors = {};
	for(const auto& gpu : EnergyManager::Hardware::GPU::getGPUs()) {
		auto monitor = std::make_shared<EnergyManager::Monitoring::GPUMonitor>(gpu, gpuMonitorInterval);
		monitors.push_back(monitor);
		gpuMonitors.push_back(monitor);
	}

	// Set up a new TestRunner
	EnergyManager::Testing::TestRunner testRunner({ std::shared_ptr<EnergyManager::Testing::Tests::ApplicationTest>(new EnergyManager::Testing::Tests::ApplicationTest(
		name,
		application,
		{
			{ "performance", "Performance= (.+?)," },
			{ "time", "Time= (.+?)," },
			{ "size", "Size= (.+?)," },
			{ "workgroupSize", "WorkgroupSize= (.+?)\n" },
		},
		applicationMonitorInterval,
		monitors)) });

	// Set up a new EnergyManager
	EnergyManager::EnergySaving::EnergyManager energyManager(
		[&] {
			// Run the tests
			testRunner.run();
		},
		{ std::make_shared<EnergyManager::EnergySaving::Strategies::DVFSStrategy>(cpus[0], gpu, dvfsStrategyInterval, dvfsModel) });

	// Run everything
	energyManager.run();
}

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	// Load the database
	const auto database = EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--database", std::string(PROJECT_DATABASE));
	EnergyManager::Persistence::Entity::initialize(database);

	// Determine monitor interval
	const auto monitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(100));
	const auto trainingMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--trainingMonitorInterval", std::chrono::milliseconds(50));

	// Define the workload
	auto cpus = EnergyManager::Hardware::CPU::parseCPUs(EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--cpus", "0"));
	auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));
	const auto sizeMultiplier = EnergyManager::Utility::Text::getArgument(arguments, "--sizeMultiplier", 50);
	const auto matrixAWidth = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixAWidth", 32 * sizeMultiplier);
	const auto matrixAHeight = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixAHeight", 32 * sizeMultiplier);
	const auto matrixBWidth = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixBWidth", 32 * sizeMultiplier);
	const auto matrixBHeight = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixBHeight", 32 * sizeMultiplier);
	auto application = EnergyManager::Utility::Application(
		std::string(CUDA_SAMPLES_DIRECTORY) + "/0_Simple/matrixMul/matrixMul",
		{ "-device=" + std::to_string(gpu->getID()),
		  "-wA=" + std::to_string(matrixAWidth),
		  "-wB=" + std::to_string(matrixBWidth),
		  "-hA=" + std::to_string(matrixAHeight),
		  "-hB=" + std::to_string(matrixBHeight) },
		cpus,
		gpu);

	// Train the model
	const auto model = train(
		application,
		EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--cpuTrainingMonitorInterval", monitorInterval),
		EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--gpuTrainingMonitorInterval", monitorInterval),
		EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--dvfsModelPath", std::string(PROJECT_RESOURCES_DIRECTORY) + "/Models/StaticDVFS"));

	// Save the model
	model->save();
	//const auto prediction = model->predict(4799979750, 4799998625, 1965000000, 1965000000);

	//return 0;

	// Test the application
	test(
		EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--name", "Matrix Multiply Test"),
		application,
		model,
		EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--dvfsStrategyInterval", std::chrono::milliseconds(100)),
		cpus,
		gpu);

	return 0;
}
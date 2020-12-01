#include <EnergyManager/EnergySaving/EnergyManager.hpp>
#include <EnergyManager/EnergySaving/Models/StaticDVFSModel.hpp>
#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Monitoring/Monitors/CPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/GPUMonitor.hpp>
#include <EnergyManager/Testing/Tests/ApplicationTest.hpp>
#include <memory>

std::shared_ptr<EnergyManager::EnergySaving::Models::DVFSModel> train(
	EnergyManager::Utility::Application& application,
	const std::chrono::system_clock::duration& cpuTrainingMonitorInterval,
	const std::chrono::system_clock::duration& gpuTrainingMonitorInterval,
	const std::string& dvfsModelPath) {
	//// Set up a model to use for the energy manager's algorithm
	//const auto dvfsModel = std::make_shared<EnergyManager::EnergySaving::Models::StaticDVFSModel>(cpuMonitor, gpuMonitor, dvfsModelPath, [&] {
	//	application.run();
	//});
	//
	//// Next, profile the application to train the model
	//dvfsModel->run();
	//
	//return dvfsModel;
}

int main(int argumentCount, char* argumentValues[]) {
	//// Parse arguments
	//const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);
	//
	//// Load the database
	//const auto database = EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--database", std::string(PROJECT_DATABASE));
	//EnergyManager::Utility::Persistence::Entity::initialize(database);
	//
	//// Determine monitor interval
	//const auto monitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(100));
	//const auto trainingMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--trainingMonitorInterval", std::chrono::milliseconds(50));
	//
	//// Define the workload
	//auto cpus = EnergyManager::Hardware::CPU::parseCPUs(EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--cpus", "0"));
	//auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));
	//const auto sizeMultiplier = EnergyManager::Utility::Text::getArgument(arguments, "--sizeMultiplier", 50);
	//const auto matrixAWidth = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixAWidth", 32 * sizeMultiplier);
	//const auto matrixAHeight = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixAHeight", 32 * sizeMultiplier);
	//const auto matrixBWidth = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixBWidth", 32 * sizeMultiplier);
	//const auto matrixBHeight = EnergyManager::Utility::Text::getArgument<unsigned long>(arguments, "--matrixBHeight", 32 * sizeMultiplier);
	//auto application = EnergyManager::Utility::Application(
	//	std::string(CUDA_SAMPLES_DIRECTORY) + "/0_Simple/matrixMul/matrixMul",
	//	{ "-device=" + std::to_string(gpu->getID()),
	//	  "-wA=" + std::to_string(matrixAWidth),
	//	  "-wB=" + std::to_string(matrixBWidth),
	//	  "-hA=" + std::to_string(matrixAHeight),
	//	  "-hB=" + std::to_string(matrixBHeight) },
	//	cpus,
	//	gpu);
	//
	//// Train the model
	//const auto model = train(
	//	application,
	//	EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--cpuTrainingMonitorInterval", monitorInterval),
	//	EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--gpuTrainingMonitorInterval", monitorInterval),
	//	EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--dvfsModelPath", std::string(PROJECT_RESOURCES_DIRECTORY) + "/Models/StaticDVFS"));
	//
	//// Save the model
	//model->save();
	//
	//return 0;
}
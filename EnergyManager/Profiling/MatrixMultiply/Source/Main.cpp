#include <EnergyManager/EnergySaving/EnergyManager.hpp>
#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Monitoring/Monitors/CPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/GPUMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/NodeMonitor.hpp>
#include <EnergyManager/Monitoring/Profiler.hpp>
#include <EnergyManager/Testing/Tests/ApplicationTest.hpp>
#include <memory>

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	// Load the database
	const auto database = EnergyManager::Utility::Text::getArgument<std::string>(arguments, "--database", std::string(PROJECT_RESOURCES_DIRECTORY) + "/Test Results/database.sqlite");
	EnergyManager::Persistence::Entity::initialize(database);

	// Add monitors
	const auto monitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(100));
	const auto applicationMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--applicationMonitorInterval", monitorInterval);
	const auto nodeMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--nodeMonitorInterval", monitorInterval);
	const auto cpuMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--cpuMonitorInterval", monitorInterval);
	const auto gpuMonitorInterval = EnergyManager::Utility::Text::getArgument<std::chrono::system_clock::duration>(arguments, "--gpuMonitorInterval", monitorInterval);
	std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::Monitor>> monitors
		= { std::make_shared<EnergyManager::Monitoring::Monitors::NodeMonitor>(EnergyManager::Hardware::Node::getNode(), nodeMonitorInterval) };
	std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::CPUMonitor>> cpuMonitors = {};
	for(const auto& cpu : EnergyManager::Hardware::CPU::getCPUs()) {
		auto monitor = std::make_shared<EnergyManager::Monitoring::Monitors::CPUMonitor>(cpu, cpuMonitorInterval);
		monitors.push_back(monitor);
		cpuMonitors.push_back(monitor);
	}
	std::vector<std::shared_ptr<EnergyManager::Monitoring::Monitors::GPUMonitor>> gpuMonitors = {};
	for(const auto& gpu : EnergyManager::Hardware::GPU::getGPUs()) {
		auto monitor = std::make_shared<EnergyManager::Monitoring::Monitors::GPUMonitor>(gpu, gpuMonitorInterval);
		monitors.push_back(monitor);
		gpuMonitors.push_back(monitor);
	}

	// Generate the profiles
	const auto sizesToTest = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--sizesToTest", 1);
	const auto sizeStart = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--sizeStart", 32 * 30);
	if(sizeStart % 32 != 0 || sizeStart < 32) {
		ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Size must be divisible by 32");
	}
	const auto sizeIncrement = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--sizeIncrement", 0);
	if(sizeIncrement % 32 != 0) {
		ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Size must be divisible by 32");
	}
	const auto cpu = EnergyManager::Hardware::CPU::getCPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--cpu", 0));
	const auto gpu = EnergyManager::Hardware::GPU::getGPU(EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));
	const auto cpuCoreClockRatesToProfile = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--cpuCoreClockRatesToProfile", 10);
	const auto gpuCoreClockRatesToProfile = EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--gpuCoreClockRatesToProfile", 10);

	const auto cpuMinimumClockRate = cpu->getMinimumCoreClockRate().toValue();
	const auto cpuMaximumClockRate = cpu->getMaximumCoreClockRate().toValue();
	const auto gpuMinimumClockRate = 1000;
	//const auto gpuMinimumClockRate = gpu->getMinimumCoreClockRate();
	const auto gpuMaximumClockRate = gpu->getMaximumCoreClockRate().toValue();
	const auto cpuClockRateIntervalSize = cpuCoreClockRatesToProfile > 1 ? (cpuMaximumClockRate - cpuMinimumClockRate) / (cpuCoreClockRatesToProfile - 1) : 0;
	const auto gpuClockRateIntervalSize = gpuCoreClockRatesToProfile > 1 ? (gpuMaximumClockRate - gpuMinimumClockRate) / (gpuCoreClockRatesToProfile - 1) : 0;
	std::vector<std::map<std::string, std::string>> profiles;
	for(unsigned int cpuClockRateIndex = 0; cpuClockRateIndex < cpuCoreClockRatesToProfile; ++cpuClockRateIndex) {
		const auto cpuClockRate = cpuMinimumClockRate + cpuClockRateIndex * cpuClockRateIntervalSize;

		for(unsigned int gpuClockRateIndex = 0; gpuClockRateIndex < gpuCoreClockRatesToProfile; ++gpuClockRateIndex) {
			const auto gpuClockRate = gpuMinimumClockRate + gpuClockRateIndex * gpuClockRateIntervalSize;

			for(unsigned int matrixSizeIndex = 0; matrixSizeIndex < sizesToTest; ++matrixSizeIndex) {
				const auto matrixSize = sizeStart + matrixSizeIndex * sizeIncrement;

				profiles.push_back({ { "cpu", EnergyManager::Utility::Text::toString(cpu->getID()) },
									 { "gpu", EnergyManager::Utility::Text::toString(gpu->getID()) },
									 { "minimumCPUClockRate", EnergyManager::Utility::Text::toString(cpuClockRate) },
									 { "maximumCPUClockRate", EnergyManager::Utility::Text::toString(cpuClockRate) },
									 { "minimumGPUClockRate", EnergyManager::Utility::Text::toString(gpuClockRate) },
									 { "maximumGPUClockRate", EnergyManager::Utility::Text::toString(gpuClockRate) },
									 { "matrixAWidth", EnergyManager::Utility::Text::toString(matrixSize) },
									 { "matrixAHeight", EnergyManager::Utility::Text::toString(matrixSize) },
									 { "matrixBWidth", EnergyManager::Utility::Text::toString(matrixSize) },
									 { "matrixBHeight", EnergyManager::Utility::Text::toString(matrixSize) } });
			}
		}
	}

	// Profile the application
	class Profiler : public EnergyManager::Monitoring::Profiler {
		using EnergyManager::Monitoring::Profiler::Profiler;

	protected:
		void beforeProfile(const std::map<std::string, std::string>& profile) final {
			const auto cpu = EnergyManager::Hardware::CPU::getCPU(std::stoi(profile.at("cpu")));
			cpu->setCoreClockRate(std::stoul(profile.at("minimumCPUClockRate")), std::stoul(profile.at("maximumCPUClockRate")));
			cpu->setTurboEnabled(false);

			const auto gpu = EnergyManager::Hardware::GPU::getGPU(std::stoi(profile.at("gpu")));
			gpu->setCoreClockRate(std::stoul(profile.at("minimumGPUClockRate")), std::stoul(profile.at("maximumGPUClockRate")));
			//gpu->setAutoBoostedClocksEnabled(false);
		}

		void onProfile(const std::map<std::string, std::string>& profile) final {
			EnergyManager::Testing::Application(
				std::string(CUDA_SAMPLES_DIRECTORY) + "/0_Simple/matrixMul/matrixMul",
				std::vector<std::string> { "-device=" + profile.at("gpu"),
										   "-wA=" + profile.at("matrixAWidth"),
										   "-wB=" + profile.at("matrixBWidth"),
										   "-hA=" + profile.at("matrixAHeight"),
										   "-hB=" + profile.at("matrixBHeight") },
				{ EnergyManager::Hardware::CPU::getCPU(std::stoi(profile.at("cpu"))) },
				EnergyManager::Hardware::GPU::getGPU(std::stoi(profile.at("gpu"))))
				.run();
		}

		void afterProfile(const std::map<std::string, std::string>& profile, const std::shared_ptr<EnergyManager::Monitoring::Persistence::ProfilerSession>& profilerSession) final {
			const auto cpu = EnergyManager::Hardware::CPU::getCPU(std::stoi(profile.at("cpu")));
			cpu->resetCoreClockRate();
			cpu->setTurboEnabled(true);

			const auto gpu = EnergyManager::Hardware::GPU::getGPU(std::stoi(profile.at("gpu")));
			gpu->resetCoreClockRate();
			//gpu->setAutoBoostedClocksEnabled(true);

			// Save the data
			profilerSession->save();
		}
	};
	Profiler profiler(
		"Matrix Multiply",
		profiles,
		monitors,
		EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--runsPerProfile", 3),
		EnergyManager::Utility::Text::getArgument<unsigned int>(arguments, "--iterationsPerRun", 1),
		true);
	profiler.run();

	return 0;
}
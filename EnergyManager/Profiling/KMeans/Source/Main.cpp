#include <EnergyManager/EnergySaving/EnergyManager.hpp>
#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Monitoring/Monitors/CPUCoreMonitor.hpp>
#include <EnergyManager/Monitoring/Profilers/FixedFrequencyProfiler.hpp>
#include <EnergyManager/Monitoring/Profilers/Profiler.hpp>
#include <EnergyManager/Testing/Tests/ApplicationTest.hpp>
#include <memory>

using EnergyManager::Hardware::CPU;
using EnergyManager::Hardware::GPU;
using EnergyManager::Monitoring::Monitors::Monitor;
using EnergyManager::Monitoring::Persistence::ProfilerSession;
using EnergyManager::Monitoring::Profilers::FixedFrequencyProfiler;
using EnergyManager::Monitoring::Profilers::Profiler;
using EnergyManager::Persistence::Entity;
using EnergyManager::Testing::Application;
using namespace EnergyManager::Utility::Text;

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	static const auto arguments = EnergyManager::Utility::Text::parseArgumentsMap(argumentCount, argumentValues);

	// Load the database
	Entity::initialize(getArgument<std::string>(arguments, "--database", std::string(PROJECT_DATABASE)));

	// Get hardware
	static const auto core = CPU::Core::getCore(getArgument<unsigned int>(arguments, "--core", 0));
	static const auto gpu = GPU::getGPU(getArgument<unsigned int>(arguments, "--gpu", 0));

	// Add monitors
	static const auto monitorInterval = getArgument<std::chrono::system_clock::duration>(arguments, "--monitorInterval", std::chrono::milliseconds(100));
	static const auto monitors = Monitor::getMonitorsForAllDevices(
		getArgument<std::chrono::system_clock::duration>(arguments, "--applicationMonitorInterval", monitorInterval),
		getArgument<std::chrono::system_clock::duration>(arguments, "--nodeMonitorInterval", monitorInterval),
		getArgument<std::chrono::system_clock::duration>(arguments, "--cpuMonitorInterval", monitorInterval),
		getArgument<std::chrono::system_clock::duration>(arguments, "--cpuCoreMonitorInterval", monitorInterval),
		getArgument<std::chrono::system_clock::duration>(arguments, "--gpuMonitorInterval", monitorInterval));

	// Generate the profiles
	std::vector<std::map<std::string, std::string>> profiles = {};
	for(const auto file : { //std::string(RODINIA_DATA_DIRECTORY) + "/kmeans/100",
							//std::string(RODINIA_DATA_DIRECTORY) + "/kmeans/204800.txt",
							//std::string(RODINIA_DATA_DIRECTORY) + "/kmeans/819200.txt",
							std::string(RODINIA_DATA_DIRECTORY) + "/kmeans/kdd_cup" }) {
		profiles.push_back({ { "core", toString(core->getID()) }, { "gpu", toString(gpu->getID()) }, /*{ "gpuSynchronizationMode", "BLOCKING" },*/ { "file", file } });
	}
	static const auto runsPerProfile = getArgument<unsigned int>(arguments, "--runsPerProfile", 1);

	// Define the workload
	static const auto workload = [&](const std::map<std::string, std::string>& profile) {
		EnergyManager::Testing::Application(std::string(RODINIA_BIN_DIRECTORY) + "/linux/cuda/kmeans", std::vector<std::string> { "-i \"" + profile.at("file") + '"' }, { core }, gpu).run();
	};

	// Check if we should generate the control data or not
	if(getArgument<bool>(arguments, "--control", true)) {
		class Profiler : public EnergyManager::Monitoring::Profilers::Profiler {
			using EnergyManager::Monitoring::Profilers::Profiler::Profiler;

		protected:
			void onProfile(const std::map<std::string, std::string>& profile) final {
				workload(profile);
			}
		};

		Profiler("KMeans", profiles, monitors, runsPerProfile, 1, true, true).run();
	} else {
		class FixedFrequencyProfiler : public EnergyManager::Monitoring::Profilers::FixedFrequencyProfiler {
			using EnergyManager::Monitoring::Profilers::FixedFrequencyProfiler::FixedFrequencyProfiler;

		protected:
			void onProfile(const std::map<std::string, std::string>& profile) final {
				workload(profile);
			}
		};

		FixedFrequencyProfiler(
			"Fixed Frequency KMeans",
			core,
			getArgument<unsigned int>(arguments, "--cpuCoreClockRatesToProfile", 5),
			gpu,
			getArgument<unsigned int>(arguments, "--gpuCoreClockRatesToProfile", 5),
			profiles,
			monitors,
			runsPerProfile,
			4,
			true,
			true)
			.run();
	}

	return 0;
}
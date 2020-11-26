#include <EnergyManager/EnergySaving/EnergyManager.hpp>
#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Monitoring/Profilers/FixedFrequencyProfiler.hpp>
#include <EnergyManager/Monitoring/Profilers/Profiler.hpp>
#include <EnergyManager/Testing/Tests/ApplicationTest.hpp>
#include <EnergyManager/Utility/Logging.hpp>
#include <memory>

using EnergyManager::Hardware::CPU;
using EnergyManager::Hardware::GPU;
using EnergyManager::Monitoring::Monitors::Monitor;
using EnergyManager::Monitoring::Persistence::ProfilerSession;
using EnergyManager::Monitoring::Profilers::FixedFrequencyProfiler;
using EnergyManager::Monitoring::Profilers::Profiler;
using EnergyManager::Persistence::Entity;
using EnergyManager::Testing::Application;
using namespace EnergyManager::Utility::Logging;
using namespace EnergyManager::Utility::Text;

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	static const auto arguments = parseArgumentsMap(argumentCount, argumentValues);
	logInformation("Program started with the following arguments: %s", join(arguments, ", ", ": ").c_str());

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
	static const auto sizeStart = getArgument<unsigned int>(arguments, "--sizeStart", 32 * 30);
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_ASSERT(sizeStart % 32 == 0 && sizeStart >= 32, "Size must be divisible by 32");
	static const auto sizeEnd = getArgument<unsigned int>(arguments, "--sizeEnd", 32 * 30);
	ENERGY_MANAGER_UTILITY_EXCEPTIONS_ASSERT(sizeEnd % 32 == 0 && sizeEnd >= sizeStart, "Size must be divisible by 32");
	std::vector<std::map<std::string, std::string>> profiles;
	for(const auto& matrixSize : Profiler::generateValueRange(sizeStart, sizeEnd, getArgument<unsigned int>(arguments, "--sizesToTest", 1))) {
		profiles.push_back({ { "core", toString(core->getID()) },
							 { "gpu", toString(gpu->getID()) },
							 /*{ "gpuSynchronizationMode", "BLOCKING" },*/
							 { "matrixAWidth", toString(matrixSize) },
							 { "matrixAHeight", toString(matrixSize) },
							 { "matrixBWidth", toString(matrixSize) },
							 { "matrixBHeight", toString(matrixSize) } });
	}
	static const auto runsPerProfile = getArgument<unsigned int>(arguments, "--runsPerProfile", 1);

	// Define the workload
	static const auto workload = [&](const std::map<std::string, std::string>& profile) {
		//		EnergyManager::Testing::Application("/bin/ping", { "-c " + std::to_string(5), "8.8.8.8" }, { core }, gpu).run();
		Application(
			std::string(CUDA_SAMPLES_DIRECTORY) + "/0_Simple/matrixMul/matrixMul",
			std::vector<std::string> { "-device=" + toString(gpu->getID()),
									   "-wA=" + profile.at("matrixAWidth"),
									   "-wB=" + profile.at("matrixBWidth"),
									   "-hA=" + profile.at("matrixAHeight"),
									   "-hB=" + profile.at("matrixBHeight") },
			{ core },
			gpu)
			.run();
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

		Profiler("Matrix Multiply", profiles, monitors, runsPerProfile, 1, true, true, true, arguments, true).run();
	} else {
		class FixedFrequencyProfiler : public EnergyManager::Monitoring::Profilers::FixedFrequencyProfiler {
			using EnergyManager::Monitoring::Profilers::FixedFrequencyProfiler::FixedFrequencyProfiler;

		protected:
			void onProfile(const std::map<std::string, std::string>& profile) final {
				workload(profile);
			}
		};

		FixedFrequencyProfiler(
			"Fixed Frequency Matrix Multiply",
			core,
			getArgument<unsigned int>(arguments, "--cpuCoreClockRatesToProfile", 1),
			gpu,
			getArgument<unsigned int>(arguments, "--gpuCoreClockRatesToProfile", 1),
			profiles,
			monitors,
			runsPerProfile,
			1,
			true,
			true,
			true,
			arguments,
			true)
			.run();
	}

	return 0;
}
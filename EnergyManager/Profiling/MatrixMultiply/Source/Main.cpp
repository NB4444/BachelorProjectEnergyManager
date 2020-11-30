#include <EnergyManager/Hardware/CPU.hpp>
#include <EnergyManager/Hardware/GPU.hpp>
#include <EnergyManager/Monitoring/Monitors/ApplicationMonitor.hpp>
#include <EnergyManager/Monitoring/Monitors/Monitor.hpp>
#include <EnergyManager/Monitoring/Profilers/Profiler.hpp>
#include <EnergyManager/Utility/Logging.hpp>
#include <memory>

using EnergyManager::Hardware::CPU;
using EnergyManager::Hardware::GPU;
using EnergyManager::Monitoring::Monitors::Monitor;
using EnergyManager::Monitoring::Persistence::ProfilerSession;
using EnergyManager::Monitoring::Profilers::Profiler;
using EnergyManager::Persistence::Entity;
using EnergyManager::Utility::Application;
using namespace EnergyManager::Utility::Logging;
using namespace EnergyManager::Utility::Text;

int main(int argumentCount, char* argumentValues[]) {
	// Parse arguments
	static const auto arguments = parseArgumentsMap(argumentCount, argumentValues);
	logInformation("Program started with the following arguments:\n%s", join(arguments, "\n", "=").c_str());

	// Load the database
	Entity::initialize(getArgument<std::string>(arguments, "--database", std::string(PROJECT_DATABASE)));

	// Get hardware
	static const auto core = CPU::Core::getCore(getArgument<unsigned int>(arguments, "--core", 0));
	static const auto gpu = GPU::getGPU(getArgument<unsigned int>(arguments, "--gpu", 0));

	// Configure and run the Profiler
	class Profiler : public EnergyManager::Monitoring::Profilers::Profiler {
		using EnergyManager::Monitoring::Profilers::Profiler::Profiler;

	protected:
		void onProfile(const std::map<std::string, std::string>& profile) final {
			//EnergyManager::Utility::Application("/bin/ping", { "-c " + std::to_string(5), "8.8.8.8" }, { core }, gpu).run();
			Application(
				std::string(CUDA_SAMPLES_DIRECTORY) + "/0_Simple/matrixMul/matrixMul",
				std::vector<std::string> { "-device=" + toString(gpu->getID()),
										   "-wA=" + profile.at("matrixAWidth"),
										   "-wB=" + profile.at("matrixBWidth"),
										   "-hA=" + profile.at("matrixAHeight"),
										   "-hB=" + profile.at("matrixBHeight") },
				{ core },
				gpu,
				true)
				.run();
		}
	} profiler(
		"Matrix Multiply",
		[&]() {
			// Get the profile configurations
			static const auto sizeStart = getArgument<unsigned int>(arguments, "--sizeStart", 4096);
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_ASSERT(sizeStart % 32 == 0 && sizeStart >= 32, "Size must be divisible by 32");
			static const auto sizeEnd = getArgument<unsigned int>(arguments, "--sizeEnd", 4096);
			ENERGY_MANAGER_UTILITY_EXCEPTIONS_ASSERT(sizeEnd % 32 == 0 && sizeEnd >= sizeStart, "Size must be divisible by 32");

			// Generate the profiles
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

			return profiles;
		}(),
		arguments);
	profiler.run();

	return 0;
}
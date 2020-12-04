#include "./ActivityTraceProfiler.hpp"

#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			void ActivityTraceProfiler::onProfile(const std::map<std::string, std::string>& profile) {
				Utility::Application(std::string(CUDA_CUPTI_SAMPLES_DIRECTORY) + "/activity_trace_async/activity_trace_async", std::vector<std::string> {}, { core_ }, gpu_, true, true).run();
			}

			ActivityTraceProfiler::ActivityTraceProfiler(const std::map<std::string, std::string>& arguments)
				: Profiler(
					"Activity Trace",
					[&]() {
						// Get hardware
						static const auto core = Hardware::CPU::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
						static const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

						// Generate the profiles
						std::vector<std::map<std::string, std::string>> profiles;
						profiles.push_back({
							{ "core", Utility::Text::toString(core->getID()) },
							{ "gpu", Utility::Text::toString(gpu->getID()) },
							/*{ "gpuSynchronizationMode", "BLOCKING" },*/
						});

						return profiles;
					}(),
					arguments)
				, core_(Hardware::CPU::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)))
				, gpu_(Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0))) {
			}
		}
	}
}

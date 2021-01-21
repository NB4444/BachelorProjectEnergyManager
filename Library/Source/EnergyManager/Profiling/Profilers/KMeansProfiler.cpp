#include "./KMeansProfiler.hpp"

#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			void KMeansProfiler::onProfile(const std::map<std::string, std::string>& profile) {
				EnergyManager::Utility::Application(std::string(RODINIA_DIRECTORY) + "/cuda/kmeans/kmeans", std::vector<std::string> { "-i", profile.at("file") }, { core_ }, gpu_, true, true).run();
			}

			KMeansProfiler::KMeansProfiler(const std::map<std::string, std::string>& arguments)
				: Profiler(
					"KMeans",
					[&]() {
						// Get hardware
						const auto core = Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
						const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

						// Generate the profiles
						std::vector<std::map<std::string, std::string>> profiles = {};
						for(const auto file : { //std::string(RODINIA_DATA_DIRECTORY) + "/kmeans/100",
												//std::string(RODINIA_DATA_DIRECTORY) + "/kmeans/204800.txt",
												//std::string(RODINIA_DATA_DIRECTORY) + "/kmeans/819200.txt",
												std::string(RODINIA_DATA_DIRECTORY) + "/kmeans/kdd_cup" }) {
							profiles.push_back({ { "core", Utility::Text::toString(core->getID()) },
												 { "gpu", Utility::Text::toString(gpu->getID()) },
												 /*{ "gpuSynchronizationMode", "BLOCKING" },*/ { "file", file } });
						}

						return profiles;
					}(),
					arguments)
				, core_(Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)))
				, gpu_(Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0))) {
			}
		}
	}
}

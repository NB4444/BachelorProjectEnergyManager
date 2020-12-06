#include "./BFSProfiler.hpp"

#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			void BFSProfiler::onProfile(const std::map<std::string, std::string>& profile) {
				Utility::Application(std::string(RODINIA_DIRECTORY) + "/cuda/bfs/bfs", std::vector<std::string> { profile.at("file") }, { core_ }, gpu_, true, true, true).run();
			}

			BFSProfiler::BFSProfiler(const std::map<std::string, std::string>& arguments)
				: Profiler(
					"BFS",
					[&]() {
						// Get hardware
						static const auto core = Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
						static const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

						// Generate the profiles
						std::vector<std::map<std::string, std::string>> profiles = {};
						for(const auto file : {
								std::string(RODINIA_DATA_DIRECTORY) + "/bfs/graph1MW_6.txt",
								//std::string(RODINIA_DATA_DIRECTORY) + "/bfs/graph4096.txt",
								//std::string(RODINIA_DATA_DIRECTORY) + "/bfs/graph65536.txt"
							}) {
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

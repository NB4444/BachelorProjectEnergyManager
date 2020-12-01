#include "./JacobiProfiler.hpp"

#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			void JacobiProfiler::onProfile(const std::map<std::string, std::string>& profile) {
				Utility::Application(
					std::string(NVIDIA_CODE_SAMPLES_DIRECTORY) + "/posts/cuda-aware-mpi-example/bin/jacobi_cuda_aware_mpi",
					std::vector<std::string> { "-d", profile.at("domainWidth"), profile.at("domainHeight") },
					{ core_ },
					gpu_)
					.run();
			}

			JacobiProfiler::JacobiProfiler(const std::map<std::string, std::string>& arguments)
				: Profiler(
					"Jacobi",
					[&]() {
						// Get hardware
						static const auto core = Hardware::CPU::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
						static const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

						// Get the profile configurations
						static const auto domainWidthStart = Utility::Text::getArgument<unsigned int>(arguments, "--domainWidthStart", 100);
						static const auto domainWidthEnd = Utility::Text::getArgument<unsigned int>(arguments, "--domainWidthEnd", 100);
						static const auto domainHeightStart = Utility::Text::getArgument<unsigned int>(arguments, "--domainHeightStart", 100);
						static const auto domainHeightEnd = Utility::Text::getArgument<unsigned int>(arguments, "--domainHeightEnd", 100);

						// Generate the profiles
						std::vector<std::map<std::string, std::string>> profiles;
						for(const auto& domainWidth : Profiler::generateValueRange(domainWidthStart, domainWidthEnd, Utility::Text::getArgument<unsigned int>(arguments, "--domainWidthsToTest", 1))) {
							for(const auto& domainHeight :
								Profiler::generateValueRange(domainWidthStart, domainWidthEnd, Utility::Text::getArgument<unsigned int>(arguments, "--domainHeightsToTest", 1))) {
								profiles.push_back({ { "core", Utility::Text::toString(core->getID()) },
													 { "gpu", Utility::Text::toString(gpu->getID()) },
													 /*{ "gpuSynchronizationMode", "BLOCKING" },*/
													 { "domainWidth", Utility::Text::toString(domainWidth) },
													 { "domainHeight", Utility::Text::toString(domainHeight) } });
							}
						}

						return profiles;
					}(),
					arguments)
				, core_(Hardware::CPU::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)))
				, gpu_(Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0))) {
			}
		}
	}
}

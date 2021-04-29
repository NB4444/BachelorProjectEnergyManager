#include "./JacobiProfiler.hpp"

#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			void JacobiProfiler::onProfile(const std::map<std::string, std::string>& profile) {
				auto application = Utility::Application(
					std::string(MPI_MPIEXEC),
					std::vector<std::string> { "--allow-run-as-root", "-n",
											   "1",
											   "--bind-to",
											   "core",
											   "--report-bindings",
											   std::string(NVIDIA_DEVELOPER_BLOG) + "/posts/cuda-aware-mpi-example/bin/jacobi_cuda_aware_mpi",
											   "-t",
											   profile.at("topologyWidth"),
											   profile.at("topologyHeight"),
											   "-d",
											   profile.at("domainWidth"),
											   profile.at("domainHeight") },
					{ core_ },
					gpu_,
					true,
					false);
				application.setGroupID(1000);
				application.setUserID(1000);

				application.run();
			}

			JacobiProfiler::JacobiProfiler(const std::map<std::string, std::string>& arguments)
				: Profiler(
					"Jacobi",
					[&]() {
						// Get hardware
						const auto core = Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
						const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

						// Get the topology configuration
						const auto topologyWidthStart = Utility::Text::getArgument<unsigned int>(arguments, "--topologyWidthStart", 1);
						const auto topologyWidthEnd = Utility::Text::getArgument<unsigned int>(arguments, "--topologyWidthEnd", 1);
						const auto topologyHeightStart = Utility::Text::getArgument<unsigned int>(arguments, "--topologyHeightStart", 1);
						const auto topologyHeightEnd = Utility::Text::getArgument<unsigned int>(arguments, "--topologyHeightEnd", 1);

						// Get the domain configuration
						const auto domainWidthStart = Utility::Text::getArgument<unsigned int>(arguments, "--domainWidthStart", 1000);
						const auto domainWidthEnd = Utility::Text::getArgument<unsigned int>(arguments, "--domainWidthEnd", 1000);
						const auto domainHeightStart = Utility::Text::getArgument<unsigned int>(arguments, "--domainHeightStart", 1000);
						const auto domainHeightEnd = Utility::Text::getArgument<unsigned int>(arguments, "--domainHeightEnd", 1000);

						// Generate the profiles
						std::vector<std::map<std::string, std::string>> profiles;
						for(const auto& topologyWidth :
							Profiler::generateValueRange(topologyWidthStart, topologyWidthEnd, Utility::Text::getArgument<unsigned int>(arguments, "--topologyWidthsToTest", 1))) {
							for(const auto& topologyHeight :
								Profiler::generateValueRange(topologyHeightStart, topologyHeightEnd, Utility::Text::getArgument<unsigned int>(arguments, "--topologyHeightsToTest", 1))) {
								for(const auto& domainWidth :
									Profiler::generateValueRange(domainWidthStart, domainWidthEnd, Utility::Text::getArgument<unsigned int>(arguments, "--domainWidthsToTest", 1))) {
									for(const auto& domainHeight :
										Profiler::generateValueRange(domainHeightStart, domainHeightEnd, Utility::Text::getArgument<unsigned int>(arguments, "--domainHeightsToTest", 1))) {
										profiles.push_back({ { "core", Utility::Text::toString(core->getID()) },
															 { "gpu", Utility::Text::toString(gpu->getID()) },
															 /*{ "gpuSynchronizationMode", "BLOCKING" },*/
															 { "topologyWidth", Utility::Text::toString(topologyWidth) },
															 { "topologyHeight", Utility::Text::toString(topologyHeight) },
															 { "domainWidth", Utility::Text::toString(domainWidth) },
															 { "domainHeight", Utility::Text::toString(domainHeight) } });
									}
								}
							}
						}

						return profiles;
					}(),
					arguments)
				, core_(Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)))
				, gpu_(Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0))) {
					setIterationsPerRun(3);
					setRunsPerProfile(1);
					setRandomize(false);
			}
		}
	}
}

#include "./MatrixMultiplyProfiler.hpp"

#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			void MatrixMultiplyProfiler::onProfile(const std::map<std::string, std::string>& profile) {
				Utility::Application(
					std::string(CUDA_SAMPLES_DIRECTORY) + "/0_Simple/matrixMul/matrixMul",
					std::vector<std::string> { "-device=" + Utility::Text::toString(gpu_->getID()),
											   "-wA=" + profile.at("matrixAWidth"),
											   "-wB=" + profile.at("matrixBWidth"),
											   "-hA=" + profile.at("matrixAHeight"),
											   "-hB=" + profile.at("matrixBHeight") },
					{ core_ },
					gpu_,
					true,
					true,
					true)
					.run();
			}

			MatrixMultiplyProfiler::MatrixMultiplyProfiler(const std::map<std::string, std::string>& arguments)
				: Profiler(
					"Matrix Multiply",
					[&]() {
						// Get hardware
						static const auto core = Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
						static const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

						// Get the profile configurations
						static const auto sizeStart = Utility::Text::getArgument<unsigned int>(arguments, "--sizeStart", 1024);
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_ASSERT(sizeStart % 32 == 0 && sizeStart >= 32, "Size must be divisible by 32");
						static const auto sizeEnd = Utility::Text::getArgument<unsigned int>(arguments, "--sizeEnd", 1024);
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_ASSERT(sizeEnd % 32 == 0 && sizeEnd >= sizeStart, "Size must be divisible by 32");

						// Generate the profiles
						std::vector<std::map<std::string, std::string>> profiles;
						for(const auto& matrixSize : Profiler::generateValueRange(sizeStart, sizeEnd, Utility::Text::getArgument<unsigned int>(arguments, "--sizesToTest", 1))) {
							profiles.push_back({ { "core", Utility::Text::toString(core->getID()) },
												 { "gpu", Utility::Text::toString(gpu->getID()) },
												 /*{ "gpuSynchronizationMode", "BLOCKING" },*/
												 { "matrixAWidth", Utility::Text::toString(matrixSize) },
												 { "matrixAHeight", Utility::Text::toString(matrixSize) },
												 { "matrixBWidth", Utility::Text::toString(matrixSize) },
												 { "matrixBHeight", Utility::Text::toString(matrixSize) } });
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

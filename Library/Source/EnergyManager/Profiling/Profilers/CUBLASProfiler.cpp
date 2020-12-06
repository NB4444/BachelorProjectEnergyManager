#include "./CUBLASProfiler.hpp"

#include "EnergyManager/Utility/Application.hpp"
#include "EnergyManager/Utility/Text.hpp"

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			void CUBLASProfiler::onProfile(const std::map<std::string, std::string>& profile) {
				Utility::Application(std::string(CUDA_SAMPLES_DIRECTORY) + "/7_CUDALibraries/simpleCUBLAS/simpleCUBLAS", {}, { core_ }, gpu_, true, true, true).run();
			}

			CUBLASProfiler::CUBLASProfiler(const std::map<std::string, std::string>& arguments)
				: Profiler(
					"CUBLAS",
					[&]() {
						// Get hardware
						static const auto core = Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0));
						static const auto gpu = Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0));

						// Generate the profiles
						std::vector<std::map<std::string, std::string>> profiles = { { { "core", Utility::Text::toString(core->getID()) }, { "gpu", Utility::Text::toString(gpu->getID()) } } };

						return profiles;
					}(),
					arguments)
				, core_(Hardware::Core::getCore(Utility::Text::getArgument<unsigned int>(arguments, "--core", 0)))
				, gpu_(Hardware::GPU::getGPU(Utility::Text::getArgument<unsigned int>(arguments, "--gpu", 0))) {
			}
		}
	}
}

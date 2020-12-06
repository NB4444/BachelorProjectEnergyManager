#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Hardware/Core.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Profiling/Profilers/Profiler.hpp"

#include <memory>

namespace EnergyManager {
	namespace Profiling {
		namespace Profilers {
			/**
			 * Profiles the Rodinia BFS application.
			 */
			class BFSProfiler : public Profiler {
				using Profiler::Profiler;

				/**
				 * The core to use when profiling.
				 */
				std::shared_ptr<Hardware::Core> core_;

				/**
				 * The GPU to use when profiling.
				 */
				std::shared_ptr<Hardware::GPU> gpu_;

			protected:
				void onProfile(const std::map<std::string, std::string>& profile) final;

			public:
				/**
				 * Creates a new BFSProfiler.
				 * @param arguments The command line arguments to use.
				 */
				explicit BFSProfiler(const std::map<std::string, std::string>& arguments);
			};
		}
	}
}
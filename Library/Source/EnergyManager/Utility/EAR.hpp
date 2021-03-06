#pragma once

#include "EnergyManager/Hardware/Core.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Units/Hertz.hpp"

#ifdef EAR_ENABLED
extern "C" {
	#include <ear.h>
}
#endif

#include <vector>

namespace EnergyManager {
	namespace Utility {
		namespace EAR {
			static void setCoreClockRates(const std::vector<unsigned int>& cores, const Utility::Units::Hertz& clockRate) {
#ifdef EAR_ENABLED
				ear_connect();

				// Encode an affinity mask
				cpu_set_t mask;
				CPU_ZERO(&mask);
				for(const auto& core : cores) {
					CPU_SET(core, &mask);
				}

				Utility::Logging::logDebug("Setting EAR CPU frequency...");

				int status = ear_set_cpufreq(&mask, clockRate.convertPrefix(Utility::Units::SIPrefix::KILO));
				if(status != EAR_SUCCESS) {
					Utility::Logging::logWarning("Could not set EAR CPU frequency: %d", status);
				}

				ear_disconnect();
#endif
			}

			static void setCoreClockRates(const std::vector<std::shared_ptr<Hardware::Core>>& cores, const Utility::Units::Hertz& clockRate) {
				std::vector<unsigned int> coreIDs;
				for(const auto& core : cores) {
					coreIDs.push_back(core->getID());
				}

				setCoreClockRates(coreIDs, clockRate);
			}

			static void setGPUClockRate(const unsigned int& gpu, const Utility::Units::Hertz& clockRate) {
#ifdef EAR_ENABLED
				ear_connect();

				Utility::Logging::logDebug("Setting EAR GPU frequency...");

				int status = ear_set_gpufreq(static_cast<int>(gpu), clockRate.convertPrefix(Utility::Units::SIPrefix::KILO));
				if(status != EAR_SUCCESS && status != -21) {
					Utility::Logging::logWarning("Could not set EAR GPU frequency: %d", status);
				}

				ear_disconnect();
#endif
			}

			static void setGPUClockRate(const std::shared_ptr<Hardware::GPU>& gpu, const Utility::Units::Hertz& clockRate) {
				setGPUClockRate(gpu->getID(), clockRate);
			}
		}
	}
}
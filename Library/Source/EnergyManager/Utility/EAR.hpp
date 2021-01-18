#pragma once

#include "EnergyManager/Hardware/Core.hpp"
#include "EnergyManager/Hardware/GPU.hpp"
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
			static void setCoreClockRates(const std::vector<std::shared_ptr<Hardware::Core>>& cores, const Utility::Units::Hertz& clockRate) {
#ifdef EAR_ENABLED
				ear_connect();

				// Encode an affinity mask
				cpu_set_t mask;
				CPU_ZERO(&mask);
				for(const auto& core : cores) {
					CPU_SET(core->getID(), &mask);
				}

				ear_set_cpufreq(&mask, clockRate.convertPrefix(Utility::Units::SIPrefix::MEGA));

				ear_disconnect();
#endif
			}

			static void setGPUClockRate(const std::shared_ptr<Hardware::GPU>& gpu, const Utility::Units::Hertz& clockRate) {
#ifdef EAR_ENABLED
				ear_connect();

				ear_set_gpufreq(static_cast<int>(gpu->getID()), clockRate.convertPrefix(Utility::Units::SIPrefix::MEGA));

				ear_disconnect();
#endif
			}
		}
	}
}
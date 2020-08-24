#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Profiling/ProcessorMonitor.hpp"

#include <map>
#include <memory>

namespace EnergyManager {
	namespace Profiling {
		class CPUMonitor :
			public ProcessorMonitor {
				std::shared_ptr<Hardware::CPU> cpu_;

				double startUserTimespan_ = 0;

				std::map<unsigned int, double> startCoreUserTimespans_ = {};

				double startNiceTimespan_ = 0;

				std::map<unsigned int, double> startCoreNiceTimespans_ = {};

				double startSystemTimespan_ = 0;

				std::map<unsigned int, double> startCoreSystemTimespans_ = {};

				double startIdleTimespan_ = 0;

				std::map<unsigned int, double> startCoreIdleTimespans_ = {};

				double startIOWaitTimespan_ = 0;

				std::map<unsigned int, double> startCoreIOWaitTimespans_ = {};

				double startInterruptsTimespan_ = 0;

				std::map<unsigned int, double> startCoreInterruptsTimespans_ = {};

				double startSoftInterruptsTimespan_ = 0;

				std::map<unsigned int, double> startCoreSoftInterruptsTimespans_ = {};

				double startStealTimespan_ = 0;

				std::map<unsigned int, double> startCoreStealTimespans_ = {};

				double startGuestTimespan_ = 0;

				std::map<unsigned int, double> startCoreGuestTimespans_ = {};

				double startGuestNiceTimespan_ = 0;

				std::map<unsigned int, double> startCoreGuestNiceTimespans_ = {};

				bool startTimespansMeasured_ = false;

			protected:
				std::map<std::string, std::string> onPoll() override;

			public:
				CPUMonitor(const std::shared_ptr<Hardware::CPU>& cpu);
		};
	}
}
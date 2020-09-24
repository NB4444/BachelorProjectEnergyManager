#pragma once

#include "EnergyManager/Hardware/CPU.hpp"
#include "EnergyManager/Monitoring/ProcessorMonitor.hpp"

#include <map>
#include <memory>

namespace EnergyManager {
	namespace Monitoring {
		class CPUMonitor : public ProcessorMonitor {
			std::shared_ptr<Hardware::CPU> cpu_;

			std::chrono::system_clock::duration startUserTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreUserTimespans_ = {};

			std::chrono::system_clock::duration startNiceTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreNiceTimespans_ = {};

			std::chrono::system_clock::duration startSystemTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreSystemTimespans_ = {};

			std::chrono::system_clock::duration startIdleTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreIdleTimespans_ = {};

			std::chrono::system_clock::duration startIOWaitTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreIOWaitTimespans_ = {};

			std::chrono::system_clock::duration startInterruptsTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreInterruptsTimespans_ = {};

			std::chrono::system_clock::duration startSoftInterruptsTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreSoftInterruptsTimespans_ = {};

			std::chrono::system_clock::duration startStealTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreStealTimespans_ = {};

			std::chrono::system_clock::duration startGuestTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreGuestTimespans_ = {};

			std::chrono::system_clock::duration startGuestNiceTimespan_;

			std::map<unsigned int, std::chrono::system_clock::duration> startCoreGuestNiceTimespans_ = {};

			bool startTimespansMeasured_ = false;

		protected:
			std::map<std::string, std::string> onPoll() override;

		public:
			static void initialize();

			CPUMonitor(const std::shared_ptr<Hardware::CPU>& cpu);
		};
	}
}
#include "./Processor.hpp"

namespace EnergyManager {
	namespace Hardware {
		void Processor::setCoreClockRate(const Utility::Units::Hertz& rate) {
			setCoreClockRate(rate, rate);
		}
	}
}
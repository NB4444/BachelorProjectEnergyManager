#include "./Processor.hpp"

namespace EnergyManager {
	namespace Hardware {
		Processor::Processor(const unsigned int& id) : id_(id) {
		}

		unsigned int Processor::getID() const {
			return id_;
		}

		void Processor::setCoreClockRate(const Utility::Units::Hertz& rate) {
			setCoreClockRate(rate, rate);
		}
	}
}
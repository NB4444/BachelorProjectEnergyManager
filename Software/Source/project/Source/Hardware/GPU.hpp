#pragma once

#include "Hardware/Device.hpp"

#include <cupti.h>
#include <functional>

namespace Hardware {
	/**
	 * Represents a Graphics Processing Unit.
	 */
	class GPU : public Device {
		void cuptiCall(const std::function<CUptiResult()>& call) const;

	public:
		GPU();

		uint32_t getTemperature() const;
	};
}
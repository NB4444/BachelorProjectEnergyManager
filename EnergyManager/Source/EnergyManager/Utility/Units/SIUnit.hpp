#pragma once

#include "EnergyManager/Utility/Units/Unit.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			enum class SIPrefix {
					YOCTO = -24,
					ZEPTO = -21,
					ATTO = -18,
					FEMTO = -15,
					PICO = -12,
					NANO = -9,
					MICRO = -6,
					MILLI = -3,
					CENTI = -2,
					DECI = -1,
					NONE = 0,
					DEKA = 1,
					HECTO = 2,
					KILO = 3,
					MEGA = 6,
					GIGA = 9,
					TERA = 12,
					PETA = 15,
					EXA = 18,
					ZETTA = 21,
					YOTTA = 24
			};

			template<typename Self, typename Type>
			class SIUnit : public Unit<Self, Type, SIPrefix> {
				public:
					using Unit<Self, Type, SIPrefix>::Unit;
			};
		}
	}
}
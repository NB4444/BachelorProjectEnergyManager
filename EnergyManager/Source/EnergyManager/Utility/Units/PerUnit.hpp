#pragma once

#include "EnergyManager/Utility/TypeTraits.hpp"

#include <string>
#include <type_traits>

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			template<typename Type, typename Per, typename Combined>
			class PerUnit {
					Type unit_;

					Per perUnit_;

					std::string perUnitName_;

					std::string perUnitSymbol_;

				public:
					PerUnit(const Type& unit, const Per& perUnit, const std::string& perUnitName = "", const std::string& perUnitSymbol = "") : unit_(unit), perUnit_(perUnit), perUnitName_(perUnitName), perUnitSymbol_(perUnitSymbol) {
					}

					Type getUnit() const {
						return unit_;
					}

					Per getPerUnit() const {
						return perUnit_;
					}

					virtual Combined toCombined() const {
						return TypeTraits::Cast<Type, Combined>::cast(unit_) / TypeTraits::Cast<Per, Combined>::cast(perUnit_);
					}

					explicit operator Combined() const {
						return toCombined();
					}

					std::string toString(const bool& symbol = false) const {
						return std::to_string(unit_) + " " + (symbol
							? perUnitSymbol_
							: (perUnitName_ == ""
								? "per " + std::to_string(perUnit_)
								: perUnitName_));
					}

					explicit operator std::string() const {
						return toString();
					}
			};
		}
	}
}
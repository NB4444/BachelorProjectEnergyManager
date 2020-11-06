#pragma once

#include "EnergyManager/Utility/TypeTraits.hpp"

#include <string>
#include <type_traits>
#include <utility>

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents a unit that represents one value per another value.
			 * @tparam Type The type of the value.
			 * @tparam Per The type of the per value.
			 * @tparam Combined The value representing a combination of the two values.
			 */
			template<typename Type, typename Per, typename Combined>
			class PerUnit {
				/**
				 * The type of the Unit value.
				 */
				Type unit_;

				/**
				 * The type of the per Unit value.
				 */
				Per perUnit_;

				/**
				 * The name of the per Unit.
				 */
				std::string perUnitName_;

				/**
				 * The symbol of the per Unit.
				 */
				std::string perUnitSymbol_;

			public:
				/**
				 * Creates a new PerUnit.
				 * @param unit The Unit.
				 * @param perUnit The per Unit.
				 * @param perUnitName The name of the per Unit.
				 * @param perUnitSymbol The symbol of the per Unit.
				 */
				PerUnit(const Type& unit, const Per& perUnit, std::string perUnitName = "", std::string perUnitSymbol = "")
					: unit_(unit)
					, perUnit_(perUnit)
					, perUnitName_(std::move(perUnitName))
					, perUnitSymbol_(std::move(perUnitSymbol)) {
				}

				/**
				 * Gets the Unit.
				 * @return The Unit.
				 */
				Type getUnit() const {
					return unit_;
				}

				/**
				 * Gets the per Unit.
				 * @return The per Unit.
				 */
				Per getPerUnit() const {
					return perUnit_;
				}

				/**
				 * Gets the combined value.
				 * @return The combined value.
				 */
				virtual Combined toCombined() const {
					return TypeTraits::Cast<Type, Combined>::cast(unit_) / TypeTraits::Cast<Per, Combined>::cast(perUnit_);
				}

				/**
				 * Gets the combined value.
				 * @return The combined value.
				 */
				explicit operator Combined() const {
					return toCombined();
				}

				/**
				 * Gets a string representation of the value.
				 * @param symbol Whether to use the symbol.
				 * @return A string representation.
				 */
				std::string toString(const bool& symbol = false) const {
					return std::to_string(unit_) + " " + (symbol ? perUnitSymbol_ : (perUnitName_.empty() ? "per " + std::to_string(perUnit_) : perUnitName_));
				}

				/**
				 * Gets a string representation of the value.
				 * @return A string representation.
				 */
				explicit operator std::string() const {
					return toString();
				}
			};
		}
	}
}
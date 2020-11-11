#pragma once

#include <cmath>
#include <string>
#include <utility>

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			/**
			 * Represents a unit.
			 * @tparam Self The type of the class.
			 * @tparam Type The type of the unit.
			 * @tparam Prefix The unit prefix type.
			 */
			template<typename Self, typename Type, typename Prefix>
			class Unit {
				/**
				 * The name.
				 */
				std::string name_;

				/**
				 * The symbol.
				 */
				std::string symbol_;

				/**
				 * The value.
				 */
				Type value_;

			public:
				/**
				 * Creates a new Unit.
				 * @param name The name.
				 * @param symbol The symbol.
				 * @param prefix The prefix.
				 */
				Unit(std::string name, std::string symbol, const Type& value, const Prefix& prefix = Prefix::NONE)
					: name_(std::move(name))
					, symbol_(std::move(symbol))
					, value_(value * std::pow(10, static_cast<int>(prefix))) {
				}

				/**
				 * Converts the value to another prefix.
				 * @param target The target prefix.
				 * @return The raw converted value.
				 */
				Type convertPrefix(const Prefix& target) const {
					return value_ / std::pow(10, static_cast<int>(target));
				}

				/**
				 * Extracts the value.
				 * @return The value.
				 */
				Type toValue() const {
					return value_;
				}

				/**
				 * Extracts the value.
				 * @return The value.
				 */
				explicit operator Type() const {
					return toValue();
				}

				/**
				 * Gets a string representation of the value.
				 * @param symbol Whether to use the symbol.
				 * @return A string representation.
				 */
				std::string toString(const bool& numeric = false, const bool& symbol = false) const {
					return std::to_string(value_) + (numeric ? "" : (" " + (symbol ? symbol_ : name_)));
				}

				/**
				 * Gets a string representation of the value.
				 * @return A string representation.
				 */
				explicit operator std::string() const {
					return toString(true, false);
				}

				/**
				 * Adds a raw value to this Unit.
				 * @param other The raw value to add.
				 * @return The resulting Unit.
				 */
				Unit& operator+=(const Type& other) {
					value_ += other;

					return *this;
				}

				/**
				 * Adds another Unit to this one.
				 * @param other The Unit to add.
				 * @return The resulting Unit.
				 */
				Unit& operator+=(const Self& other) {
					value_ += other.value_;

					return *this;
				}

				/**
				 * Adds two Units.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return The resulting Unit.
				 */
				friend Self operator+(const Self& left, const Self& right) {
					auto result = left;
					result += right;

					return result;
				}

				/**
				 * Subtracts a raw value from this Unit.
				 * @param other The raw value to subtract.
				 * @return The resulting Unit.
				 */
				Unit& operator-=(const Type& other) {
					value_ -= other;

					return *this;
				}

				/**
				 * Subtracts a Unit from this one.
				 * @param other The Unit to subtract.
				 * @return The resulting Unit.
				 */
				Unit& operator-=(const Self& other) {
					value_ -= other.value_;

					return *this;
				}

				/**
				 * Subtracts a Unit from another one.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return The resulting Unit.
				 */
				friend Self operator-(const Self& left, const Self& right) {
					auto result = left;
					result -= right;

					return result;
				}

				/**
				 * Multiplies a raw value with this Unit.
				 * @param other The raw value to multiply.
				 * @return The resulting Unit.
				 */
				Unit& operator*=(const Type& other) {
					value_ *= other;

					return *this;
				}

				/**
				 * Multiplies another Unit with this one.
				 * @param other The unit to multiply.
				 * @return The resulting Unit.
				 */
				Unit& operator*=(const Self& other) {
					value_ *= other.value_;

					return *this;
				}

				/**
				 * Multiplies two Units.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return The resulting Unit.
				 */
				friend Self operator*(const Self& left, const Self& right) {
					auto result = left;
					result *= right;

					return result;
				}

				/**
				 * Divides this Unit by a raw value.
				 * @param other The raw value to divide by.
				 * @return The resulting Unit.
				 */
				Unit& operator/=(const Type& other) {
					value_ /= other;

					return *this;
				}

				/**
				 * Divides this Unit by another one.
				 * @param other The Unit to divide by.
				 * @return The resulting Unit.
				 */
				Unit& operator/=(const Self& other) {
					value_ /= other.value_;

					return *this;
				}

				/**
				 * Divides one Unit by another one.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return The resulting Unit.
				 */
				friend Self operator/(const Self& left, const Self& right) {
					auto result = left;
					result /= right;

					return result;
				}

				/**
				 * Determines if one Unit is larger than another one.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return Whether the left Unit is larger.
				 */
				friend bool operator>(const Self& left, const Self& right) {
					return left.toValue() > right.toValue();
				}

				/**
				 * Determines if one Unit is larger than or equal to another one.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return Whether the left Unit is larger or equal.
				 */
				friend bool operator>=(const Self& left, const Self& right) {
					return left.toValue() >= right.toValue();
				}

				/**
				 * Determines if one Unit is smaller than another one.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return Whether the left Unit is smaller.
				 */
				friend bool operator<(const Self& left, const Self& right) {
					return left.toValue() < right.toValue();
				}

				/**
				 * Determines if one Unit is smaller than or equal to another one.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return Whether the left Unit is smaller or equal.
				 */
				friend bool operator<=(const Self& left, const Self& right) {
					return left.toValue() <= right.toValue();
				}

				/**
				 * Determines if two Units are equal.
				 * @param left The left operand Unit.
				 * @param right The right operand Unit.
				 * @return Whether the Units are equal.
				 */
				friend bool operator==(const Self& left, const Self& right) {
					return left.toValue() == right.toValue();
				}
			};
		}
	}
}
#pragma once

#include <string>
#include <cmath>

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			template<typename Self, typename Type, typename Prefix>
			class Unit {
					std::string name_;

					std::string symbol_;

					Type value_;

				public:
					Unit(const std::string& name, const std::string& abbreviation, const Type& value, const Prefix& prefix = Prefix::NONE) : name_(name), value_(value * std::pow(10, static_cast<int>(prefix))) {
					}

					Type convertPrefix(const Prefix& target) const {
						return value_ / std::pow(10, static_cast<int>(target));
					}

					Type toValue() const {
						return value_;
					}

					explicit operator Type() const {
						return toValue();
					}

					std::string toString(const bool& symbol = false) const {
						return std::to_string(value_) + " " + (symbol
							? symbol_
							: name_);
					}

					explicit operator std::string() const {
						return toString();
					}

					Unit& operator +=(const Type& other) {
						value_ += other;

						return *this;
					}

					Unit& operator +=(const Self& other) {
						value_ += other.value_;

						return *this;
					}

					friend Self operator +(const Self& left, const Self& right) {
						auto result = left;
						result += right;

						return result;
					}

					Unit& operator -=(const Type& other) {
						value_ -= other;

						return *this;
					}

					Unit& operator -=(const Self& other) {
						value_ -= other.value_;

						return *this;
					}

					friend Self operator -(const Self& left, const Self& right) {
						auto result = left;
						result -= right;

						return result;
					}

					Unit& operator *=(const Type& other) {
						value_ *= other;

						return *this;
					}

					Unit& operator *=(const Self& other) {
						value_ *= other.value_;

						return *this;
					}

					friend Self operator *(const Self& left, const Self& right) {
						auto result = left;
						result *= right;

						return result;
					}

					Unit& operator /=(const Type& other) {
						value_ /= other;

						return *this;
					}

					Unit& operator /=(const Self& other) {
						value_ /= other.value_;

						return *this;
					}

					friend Self operator /(const Self& left, const Self& right) {
						auto result = left;
						result /= right;

						return result;
					}

					friend bool operator >(const Self& left, const Self& right) {
						return left.toValue() > right.toValue();
					}

					friend bool operator >=(const Self& left, const Self& right) {
						return left.toValue() >= right.toValue();
					}

					friend bool operator <(const Self& left, const Self& right) {
						return left.toValue() < right.toValue();
					}

					friend bool operator <=(const Self& left, const Self& right) {
						return left.toValue() <= right.toValue();
					}

					friend bool operator ==(const Self& left, const Self& right) {
						return left.toValue() == right.toValue();
					}
			};
		}
	}
}
#include "./Bandwidth.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Units {
			Bandwidth::Bandwidth(const Byte& value, const std::chrono::system_clock::duration& duration) : PerUnit(value, duration) {
			}

			double Bandwidth::toCombined() const {
				return static_cast<double>(getUnit().toValue()) / (std::chrono::duration_cast<std::chrono::milliseconds>(getPerUnit()).count() / 1000.0);
			}
		}
	}
}
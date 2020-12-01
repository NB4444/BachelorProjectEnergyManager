#pragma once

#include "EnergyManager/Utility/Units/Byte.hpp"

#include <chrono>
#include <type_traits>

namespace EnergyManager {
	namespace Utility {
		namespace TypeTraits {
			/**
			 * Checks whether the specified type is a duration type.
			 * @tparam Type The type.
			 */
			template<class Type>
			struct IsDuration : std::false_type {};

			template<class Representation, class Period>
			struct IsDuration<std::chrono::duration<Representation, Period>> : std::true_type {};

			/**
			 * Casts a value.
			 * @tparam Source The source type.
			 * @tparam Target The target type.
			 */
			template<typename Source, typename Target>
			struct Cast {
				static Target cast(const Source& value) {
					return static_cast<Target>(value);
				}
			};

			template<typename Target, typename Representation, typename Period>
			struct Cast<std::chrono::duration<Representation, Period>, Target> {
				static Target cast(const std::chrono::duration<Representation, Period>& value) {
					return static_cast<Target>(value.count());
				}
			};

			template<typename Target>
			struct Cast<Units::Byte, Target> {
				static Target cast(const Units::Byte& value) {
					return static_cast<Target>(value.toValue());
				}
			};
		}
	}
}
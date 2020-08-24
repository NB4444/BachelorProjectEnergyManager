#pragma once

#include <type_traits>

namespace EnergyManager {
	namespace Utility {
		namespace TypeTraits {
			template<typename Type, template<typename...> class Ref>
			struct IsSpecialization :
				std::false_type {
			};

			template<template<typename...> class Ref, typename... Args>
			struct IsSpecialization<Ref<Args...>, Ref> :
				std::true_type {
			};
		}
	}
}
#pragma once

#include "Testing/Test.hpp"

namespace Testing {
	/**
	 * Represents the results of a single Test.
	 */
	class TestResults {
		/**
		 * The Test associated with the results.
		 */
		Test test_;

	public:
		TestResults();
	};
}
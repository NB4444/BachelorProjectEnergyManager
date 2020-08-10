#pragma once

#include "Test.hpp"

namespace EnergyManager {
	namespace Testing {
		class TestResults;

		namespace Tests {
			class ApplicationTest : public Test {
				/**
				 * The Application to test.
				 */
				Application application_;

				/**
				 * The parameters to use to run the Application.
				 */
				std::vector<std::string> parameters_;

				/**
				 * The results to parse from the Application's output.
				 */
				std::map<std::string, std::string> results_;

				TestResults onRun() override;

			public:
				/**
				 * Creates a new ApplicationTest.
				 * @param name The name of the ApplicationTest.
				 * @param application The Application to test.
				 * @param parameters The parameters to use to run the Application.
				 * @param results The results to parse from the Application's output.
				 */
				ApplicationTest(const std::string& name, const Application& application, std::vector<std::string> parameters, std::map<std::string, std::string> results);
			};
		}
	}
}
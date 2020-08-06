#pragma once

#include "Application.hpp"
#include "Persistence/Entity.hpp"

namespace Testing {
	class TestResults;

	/**
	 * A test of an Application.
	 */
	class Test : public Persistence::Entity<Test> {
		/**
		 * The name of the Test.
		 */
		std::string name_;

		/**
		 * The Application to test.
		 */
		Application application_;

		/**
		 * The parameters to provide to the Application.
		 */
		std::vector<std::string> parameters_;

		/**
		 * The results to parse, identified by their name and an associated regular expression.
		 */
		std::map<std::string, std::string> results_;

		std::map<std::string, std::string> onSave() override;

	public:
		Test(const std::map<std::string, std::string>& row);

		/**
		 * Creates a new Test.
		 * @param name The name of the Test.
		 * @param application The Application to test.
		 * @param parameters The parameters to provide to the Application.
		 * @param results The results to parse, identified by their name and an associated regular expression.
		 */
		Test(std::string name, const Application& application, std::vector<std::string> parameters, std::map<std::string, std::string> results);

		/**
		 * Gets the name of the Test.
		 * @return The name.
		 */
		std::string getName() const;

		/**
		 * Gets the Application to Test.
		 * @return The Application.
		 */
		Application getApplication() const;

		/**
		 * Gets the parameters to provide to the Application.
		 * @return The parameters.
		 */
		std::vector<std::string> getParameters() const;

		/**
		 * Gets the results to parse.
		 * @return The results.
		 */
		std::map<std::string, std::string> getResults() const;

		/**
		 * Runs the Test.
		 * @return The parsed Test results.
		 */
		TestResults run();
	};
}
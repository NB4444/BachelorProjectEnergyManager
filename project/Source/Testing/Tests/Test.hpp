#pragma once

#include "Application.hpp"
#include "Persistence/Entity.hpp"

namespace Testing {
	class TestResults;

	namespace Tests {
		/**
		 * A test of an Application.
		 */
		class Test : public Persistence::Entity<Test> {
			/**
			 * The name of the Test.
			 */
			std::string name_;

			std::map<std::string, std::string> onSave() override;

		protected:
			/**
			 * Executes the Test.
			 * @return The results of the Test.
			 */
			virtual TestResults onRun();

		public:
			Test(const std::map<std::string, std::string>& row);

			/**
			 * Creates a new Test.
			 * @param name The name of the Test.
			 * @param
			 */
			Test(std::string name);

			/**
			 * Gets the name of the Test.
			 * @return The name.
			 */
			std::string getName() const;

			/**
			 * Runs the Test.
			 * @return The parsed Test results.
			 */
			TestResults run();
		};
	}
}
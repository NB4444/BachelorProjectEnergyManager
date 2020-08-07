#pragma once

#include "Utility/Logging.hpp"
#include "Utility/Serialization.hpp"
#include "Utility/Text.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <sqlite3.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Persistence {
	/**
	 * A persistent object.
	 * @tparam Type The type of the Entity class.
	 */
	template<typename Type>
	class Entity {
		/**
		 * The database to use.
		 */
		static sqlite3* database_;

		/**
		 * The database ID of the Entity.
		 */
		int id_ = -1;

		/**
		 * The name of the Entity.
		 */
		std::string name_;

		/**
		 * Executes the SQL statement.
		 * @param statement The statement to execute.
		 * @param callback The callback that is called with the result.
		 */
		void executeSQL(
			const std::string& statement,
			const std::function<int(void*, int, char**, char**)>& callback = [](void* unused, int argumentCount, char** argumentValues, char** columnNames) -> int {
				return 0;
			}) {
			char* errorMessage = nullptr;

			//// Prepare the statement
			//sqlite3_stmt* preparedStatement;
			//sqlite3_prepare(database_, statement.c_str(), sizeof(statement.c_str()), &preparedStatement, nullptr);

			// Execute the statement
			//bool done = false;
			//for(int row = 0; !done; ++row) {
			//	printf("In select while\n");
			//	int errorCode;
			//	switch(errorCode = sqlite3_step(preparedStatement)) {
			//		case SQLITE_ROW: {
			//			size_t bytes = sqlite3_column_bytes(preparedStatement, 0);
			//			const unsigned char* text = sqlite3_column_text(preparedStatement, 1);
			//			printf("count %d: %s (%d bytes)\n", row, text, bytes);
			//			break;
			//		}
			//		case SQLITE_DONE: {
			//			done = true;
			//			break;
			//		}
			//		default: {
			//			throw std::runtime_error("Could not execute SQL statement " + statement + ": " + (errorMessage == nullptr ? std::to_string(errorCode) : std::string(errorMessage)));
			//		}
			//	}
			//}
			//
			//sqlite3_finalize(preparedStatement);
			int errorCode;
			if(errorCode = sqlite3_exec(database_, statement.c_str(), callback.target<int(void*, int, char**, char**)>(), nullptr, &errorMessage)) {
				throw std::runtime_error("Could not execute SQL statement " + statement + ": " + (errorMessage == nullptr ? std::to_string(errorCode) : std::string(errorMessage)));
			}
		}

	protected:
		/**
		 * Called when data is saved to the database.
		 * Used to store the object's fields.
		 * @return The database row that should be saved.
		 */
		virtual std::map<std::string, std::string> onSave() {
			return {};
		}

	public:
		static Type load(const int& id) {
			// TODO: Retrieve row from database as map of strings to strings and pass it to the load function below
			return Type(std::map<std::string, std::string> {});
		}

		/**
		 * Creates a new Entity.
		 * @param name The name of the Entity.
		 */
		Entity(std::string name)
			: name_(std::move(name)) {
			// Open a new database connection
			if(database_ == nullptr && sqlite3_open("database.sqlite", &database_)) {
				throw std::runtime_error("Cannot open database: " + std::string(sqlite3_errmsg(database_)));
			}
		}

		/**
		 * Destructs the Entity.
		 */
		~Entity() {
			//if(database_ != nullptr) {
			//	sqlite3_close(database_);
			//	database_ = nullptr;
			//}
		}

		/**
		 * Gets the Application's ID.
		 * @return The ID.
		 */
		int getID() const {
			return id_;
		}

		/**
		 * Saves the Entity to the database.
		 */
		void save() {
			// Retrieve the results
			std::map<std::string, std::string> results = this->onSave();

			// Collect columns
			std::vector<std::string> columns;
			for(const auto& result : results) {
				columns.push_back(result.first);
			}
			std::sort(columns.begin(), columns.end());

			// Collect values
			std::vector<std::string> values;
			for(const auto& column : columns) {
				values.push_back(results[column]);
			}

			// Create a table for the current Entity if it does not yet exist
			std::vector<std::string> columnsWithType;
			std::transform(columns.begin(), columns.end(), std::back_inserter(columnsWithType), [](const std::string& column) -> std::string {
				return column + " TEXT";
			});
			executeSQL(
				"CREATE TABLE IF NOT EXISTS " + name_ + "("
				+ "id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
				+ Utility::Text::join(columnsWithType, ",")
				+ ")");

			// Store the results
			executeSQL(
				"INSERT OR REPLACE INTO " + name_ + "("
				+ (id_ == -1 ? "" : "id,")
				+ Utility::Text::join(columns, ",")
				+ ") VALUES ("
				+ (id_ == -1 ? "" : (std::to_string(id_) + ","))
				+ Utility::Serialization::serialize(values, ",", "\"\"")
				+ ")");
			// TODO: Set correct ID if it was generated automatically

			// Test
			executeSQL(
				"SELECT * FROM " + name_ + "",
				[](void* unused, int argumentCount, char** argumentValues, char** columnNames) -> int {
					Utility::Logging::logInformation("@@@@@@@@@@@@@@@@@@@@@2");
					return 0;
				});
		}
	};

	template<typename Type>
	sqlite3* Entity<Type>::database_ = nullptr;
}
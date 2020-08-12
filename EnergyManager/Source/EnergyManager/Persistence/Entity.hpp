#pragma once

#include "EnergyManager/Utility/Logging.hpp"
#include "EnergyManager/Utility/Text.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <sqlite3.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace EnergyManager {
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
				int errorCode = sqlite3_exec(database_, statement.c_str(), callback.target<int(void*, int, char**, char**)>(), nullptr, &errorMessage);
				if(errorCode) {
					throw std::runtime_error("Could not execute SQL statement " + statement + ": " + (errorMessage == nullptr ? std::to_string(errorCode) : std::string(errorMessage)));
				}
			}

		protected:
			/**
			 * Called when data is saved to the database.
			 * Used to store the object's fields.
			 */
			virtual void onSave() {
			}

		public:
			/**
			 * Creates a new Entity.
			 * @param name The name of the Entity.
			 */
			Entity() {
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

			void insert(const std::string& table, std::vector<std::map<std::string, std::string>> rowColumnValues) {
				// Collect columns
				std::set<std::string> columns;
				for(const auto& columnValues : rowColumnValues) {
					for(const auto& columnValue : columnValues) {
						columns.insert(columnValue.first);
					}
				}

				// Collect values
				std::vector<std::vector<std::string>> values;
				for(auto& columnValues : rowColumnValues) {
					std::vector<std::string> insertValues;

					for(const auto& column : columns) {
						insertValues.push_back(columnValues[column]);
					}

					values.push_back(insertValues);
				}

				std::vector<std::string> insertRows;
				std::transform(values.begin(), values.end(), std::back_inserter(insertRows), [](const auto& item) {
					return Utility::Text::join(item, ",");
				});

				// Store the results
				executeSQL("INSERT INTO " + table + "(" + Utility::Text::join(columns, ",") + ") VALUES (" + Utility::Text::join(insertRows, "),(") + ");");
			}

			void insert(const std::string& table, std::map<std::string, std::string> columnValues) {
				insert(table, { columnValues });
			}

			void createTable(const std::string& table, const std::map<std::string, std::string>& columnsWithAttributes) {
				std::vector<std::string> columnsWithAttributesSerialized;
				std::transform(columnsWithAttributes.begin(), columnsWithAttributes.end(), std::back_inserter(columnsWithAttributesSerialized), [](const auto& column) -> std::string {
					return column.first + " " + column.second;
				});
				executeSQL("CREATE TABLE " + table + "(" + Utility::Text::join(columnsWithAttributesSerialized, ",") + ");");
			}

			void addColumn(const std::string& table, const std::string& column, const std::string& attributes) {
				executeSQL("ALTER TABLE " + table + " ADD " + column + " " + attributes + ";");
			}

			/**
			 * Saves the Entity to the database.
			 */
			void save() {
				onSave();
			}
		};

		template<typename Type>
		sqlite3* Entity<Type>::database_ = nullptr;
	}
}
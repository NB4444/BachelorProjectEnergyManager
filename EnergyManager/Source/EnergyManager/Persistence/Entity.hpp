#pragma once

#include "EnergyManager/Utility/Exception.hpp"
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

			static std::string databaseFile_;

			static std::vector<std::map<std::string, std::string>> rows_;

			static int callback(void* context, int columnCount, char** columnValues, char** columnNames) {
				rows_.clear();

				std::map<std::string, std::string> row;

				for(size_t columnIndex = 0u; columnIndex < columnCount; ++columnIndex) {
					row[columnNames[columnIndex]] = columnValues[columnIndex];
				}

				rows_.push_back(row);

				return 0;
			}

			/**
			 * Executes the SQL statement.
			 * @param statement The statement to execute.
			 * @return The rows that were returned.
			 */
			std::vector<std::map<std::string, std::string>> executeSQL(const std::string& statement) {
				// Open a new database connection
				if(database_ == nullptr && sqlite3_open(databaseFile_.c_str(), &database_)) {
					ENERGY_MANAGER_UTILITY_EXCEPTION("Cannot open database: " + std::string(sqlite3_errmsg(database_)));
				}

				// Execute the statement
				char* errorMessage = nullptr;
				int errorCode = sqlite3_exec(database_, statement.c_str(), callback, nullptr, &errorMessage);
				if(errorCode) {
					ENERGY_MANAGER_UTILITY_EXCEPTION("Could not execute SQL statement " + statement + ": " + (errorMessage == nullptr ? std::to_string(errorCode) : std::string(errorMessage)));
				}

				return rows_;
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
			 */
			Entity(const std::string& databaseFile) {
				databaseFile_ = databaseFile;
			}

			void addColumn(const std::string& table, const std::string& column, const std::string& attributes) {
				executeSQL("ALTER TABLE " + table + " ADD " + column + " " + attributes + ";");
			}

			void createTable(const std::string& table, const std::map<std::string, std::string>& columnsWithAttributes) {
				std::vector<std::string> columnsWithAttributesSerialized;
				std::transform(columnsWithAttributes.begin(), columnsWithAttributes.end(), std::back_inserter(columnsWithAttributesSerialized), [](const auto& column) -> std::string {
					return column.first + " " + column.second;
				});
				executeSQL("CREATE TABLE " + table + "(" + Utility::Text::join(columnsWithAttributesSerialized, ",") + ");");
			}

			void insert(const std::string& table, std::vector<std::map<std::string, std::string>>& rowColumnValues) {
				// Collect columns
				std::set<std::string> columns;
				for(const auto& columnValues : rowColumnValues) {
					for(const auto& columnValue : columnValues) {
						columns.insert(columnValue.first);
					}
				}

				// Collect values
				std::vector<std::vector<std::string>> rowValues;
				for(auto& columnValues : rowColumnValues) {
					std::vector<std::string> insertValues;

					for(const auto& column : columns) {
						insertValues.push_back(columnValues[column]);
					}

					rowValues.push_back(insertValues);
				}

				std::vector<std::string> insertRows;
				std::transform(rowValues.begin(), rowValues.end(), std::back_inserter(insertRows), [](const auto& item) {
					return Utility::Text::join(item, ",");
				});

				// Store the results
				executeSQL("INSERT INTO " + table + "(" + Utility::Text::join(columns, ",") + ") VALUES(" + Utility::Text::join(insertRows, "),(") + ");");
			}

			void insert(const std::string& table, const std::map<std::string, std::string>& columnValues) {
				insert(table, { columnValues });
			}

			void select(const std::string& table, const std::vector<std::string>& columns, const std::string& conditions) {
				executeSQL("SELECT " + Utility::Text::join(columns, ",") + " FROM " + table + " WHERE " + conditions + " ;");
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

		template<typename Type>
		std::string Entity<Type>::databaseFile_ = "database.sqlite";

		template<typename Type>
		std::vector<std::map<std::string, std::string>> Entity<Type>::rows_ = {};
	}
}
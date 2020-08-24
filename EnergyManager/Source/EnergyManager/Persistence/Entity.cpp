#include "./Entity.hpp"

namespace EnergyManager {
	namespace Persistence {
		sqlite3* Entity::database_ = nullptr;

		std::vector<std::map<std::string, std::string>> Entity::rows_ = {};

		int Entity::callback(void* context, int columnCount, char** columnValues, char** columnNames) {
			rows_.clear();

			std::map<std::string, std::string> row;

			for(size_t columnIndex = 0u; columnIndex < columnCount; ++columnIndex) {
				row[columnNames[columnIndex]] = columnValues[columnIndex];
			}

			rows_.push_back(row);

			return 0;
		}

		void Entity::onSave() {
		}

		void Entity::setDatabaseFile(const std::string& databaseFile) {
			// Open a new database connection
			if(database_ == nullptr && sqlite3_open(databaseFile.c_str(), &database_)) {
				ENERGY_MANAGER_UTILITY_EXCEPTION("Cannot open database: " + std::string(sqlite3_errmsg(database_)));
			}
		}

		std::vector<std::map<std::string, std::string>> Entity::executeSQL(const std::string& statement, const std::string& file, const int& line) {
			// Execute the statement
			char* errorMessage = nullptr;
			int errorCode = sqlite3_exec(database_, statement.c_str(), callback, nullptr, &errorMessage);
			if(errorCode) {
				throw EnergyManager::Utility::Exception(
					"Could not execute SQL statement " + statement + ": " + (errorMessage == nullptr ? std::to_string(errorCode) : std::string(errorMessage)),
					file,
					line);
			}

			return rows_;
		}

		void Entity::addColumn(const std::string& table, const std::string& column, const std::string& attributes) {
			ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL("ALTER TABLE " + table + " ADD " + column + " " + attributes + ";");
		}

		void Entity::createTable(const std::string& table, const std::map<std::string, std::string>& columnsWithAttributes) {
			std::vector<std::string> columnsWithAttributesSerialized;
			std::transform(columnsWithAttributes.begin(), columnsWithAttributes.end(), std::back_inserter(columnsWithAttributesSerialized), [](const auto& column) -> std::string {
				return column.first + " " + column.second;
			});
			ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL("CREATE TABLE " + table + "(" + Utility::Text::join(columnsWithAttributesSerialized, ",") + ");");
		}

		void Entity::insert(const std::string& table, std::vector<std::map<std::string, std::string>>& rowColumnValues) {
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
			ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL("INSERT INTO " + table + "(" + Utility::Text::join(columns, ",") + ") VALUES(" + Utility::Text::join(insertRows, "),(") + ");");
		}

		void Entity::insert(const std::string& table, const std::map<std::string, std::string>& columnValues) {
			insert(table, { columnValues });
		}

		void Entity::select(const std::string& table, const std::vector<std::string>& columns, const std::string& conditions) {
			ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL("SELECT " + Utility::Text::join(columns, ",") + " FROM " + table + " WHERE " + conditions + " ;");
		}

		void Entity::save() {
			onSave();
		}
	}
}
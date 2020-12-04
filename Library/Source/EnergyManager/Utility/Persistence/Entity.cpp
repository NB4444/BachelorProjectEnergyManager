#include "./Entity.hpp"

#include "EnergyManager/Utility/Collections.hpp"
#include "EnergyManager/Utility/Logging.hpp"

namespace EnergyManager {
	namespace Utility {
		namespace Persistence {
			sqlite3* Entity::database_ = nullptr;

			StaticInitializer Entity::databaseInitializer_ = StaticInitializer(
				[] {
					auto databaseFile = std::string(PROJECT_DATABASE);

					// Open a new database connection
					Utility::Logging::logDebug("Initializing database %s...", databaseFile.c_str());
					if(database_ == nullptr && sqlite3_open_v2(databaseFile.c_str(), &database_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, "unix-dotfile")) {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Cannot open database: " + std::string(sqlite3_errmsg(database_)));
					}
					//if(database_ == nullptr && sqlite3_open(databaseFile.c_str(), &database_)) {
					//	ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Cannot open database: " + std::string(sqlite3_errmsg(database_)));
					//}
				},
				[] {
					Utility::Logging::logDebug("Closing database...");
					if(database_ == nullptr && sqlite3_close(database_)) {
						ENERGY_MANAGER_UTILITY_EXCEPTIONS_EXCEPTION("Cannot close database: " + std::string(sqlite3_errmsg(database_)));
					}
				});

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

			std::vector<std::map<std::string, std::string>> Entity::executeSQL(const std::string& statement, const std::string& file, const int& line) {
				// Keep track of the state
				char* errorMessage = nullptr;
				int errorCode;

				// Keep trying the statement if the database is locked
				do {
					// Execute the statement
					errorCode = sqlite3_exec(database_, statement.c_str(), callback, nullptr, &errorMessage);
				} while(errorCode == SQLITE_BUSY);

				if(errorCode) {
					throw EnergyManager::Utility::Exceptions::Exception(
						"Could not execute SQL statement " + statement + ", error code " + std::to_string(errorCode) + (errorMessage == nullptr ? "" : ": " + std::string(errorMessage)),
						file,
						line);
				}

				return rows_;
			}

			void Entity::addColumn(const std::string& table, const std::string& column, const std::string& attributes) {
				Logging::logTrace("Adding database column %s to table %s...", column.c_str(), table.c_str());
				ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL("ALTER TABLE " + table + " ADD " + column + " " + attributes + ";");
			}

			void Entity::createTable(const std::string& table, const std::map<std::string, std::string>& columnsWithAttributes) {
				Logging::logTrace("Creating database table %s...", table.c_str());

				std::vector<std::string> columnsWithAttributesSerialized;
				std::transform(columnsWithAttributes.begin(), columnsWithAttributes.end(), std::back_inserter(columnsWithAttributesSerialized), [](const auto& column) -> std::string {
					return column.first + " " + column.second;
				});
				ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL("CREATE TABLE " + table + "(" + Utility::Text::join(columnsWithAttributesSerialized, ",") + ");");
			}

			void Entity::insert(const std::string& table, std::vector<std::map<std::string, std::string>> rowColumnValues) {
				Logging::logTrace("Inserting data in table %s...", table.c_str());

				// If there are no rows we can just exit
				if(rowColumnValues.empty()) {
					return;
				}

				// Fetch the maximum row count in one operation
				const auto compoundSelectLimit = sqlite3_limit(database_, SQLITE_LIMIT_COMPOUND_SELECT, -1);

				// Check if we need to split the current operation
				if(rowColumnValues.size() > compoundSelectLimit) {
					Utility::Logging::logTrace("Row count of %d exceeds limit of %d for one insert operation, splitting operation into chunks...", rowColumnValues.size(), compoundSelectLimit);

					// Chunk the rows
					for(const auto& currentRowColumnValues : Utility::Collections::splitInChunks(rowColumnValues, compoundSelectLimit)) {
						// Process the current chunk
						insert(table, currentRowColumnValues);
					}
				} else {
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
							auto insertValue = columnValues.at(column);
							insertValues.push_back(insertValue);
						}

						rowValues.push_back(insertValues);
					}

					std::vector<std::string> insertRows;
					std::transform(rowValues.begin(), rowValues.end(), std::back_inserter(insertRows), [](const auto& item) {
						return Utility::Text::join(item, ",");
					});

					// Store the results
					ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL(
						"INSERT OR REPLACE INTO " + table + "(" + Utility::Text::join(columns, ",") + ") VALUES(" + Utility::Text::join(insertRows, "),(") + ");");
				}
			}

			unsigned long Entity::insert(const std::string& table, const std::map<std::string, std::string>& columnValues) {
				insert(table, std::vector<std::map<std::string, std::string>> { columnValues });

				return sqlite3_last_insert_rowid(database_);
			}

			void Entity::select(const std::string& table, const std::vector<std::string>& columns, const std::string& conditions) {
				Logging::logTrace("Selecting data from table %s...", table.c_str());

				ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL("SELECT " + Utility::Text::join(columns, ",") + " FROM " + table + " WHERE " + conditions + " ;");
			}

			void Entity::createIndex(const std::string& table, const std::string& index, const std::vector<std::string>& columns, const bool& unique) {
				ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL(std::string("CREATE") + (unique ? " UNIQUE" : "") + " INDEX " + index + " ON " + table + "(" + Utility::Text::join(columns, ",") + ");");
			}

			std::string Entity::filterSQL(std::string sql) {
				const std::string find = "\"";
				size_t startIndex = sql.find(find);
				if(startIndex != std::string::npos) {
					sql.replace(startIndex, find.length(), "\"\"");
				}

				return sql;
			}

			long Entity::getID() const {
				return id_;
			}

			void Entity::setID(const long& id) {
				id_ = id;
			}

			void Entity::save() {
				onSave();
			}
		}
	}
}
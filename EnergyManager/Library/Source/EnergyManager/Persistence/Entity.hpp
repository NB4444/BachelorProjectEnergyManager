#pragma once

#include "EnergyManager/Utility/Exceptions/Exception.hpp"
#include "EnergyManager/Utility/Logging/Loggable.hpp"
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

#define ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL(STATEMENT) EnergyManager::Persistence::Entity::executeSQL(STATEMENT, __FILE__, __LINE__)

namespace EnergyManager {
	namespace Persistence {
		/**
		 * A persistent object.
		 */
		class Entity : protected Utility::Logging::Loggable {
			/**
			 * The database to use.
			 */
			static sqlite3* database_;

			/**
			 * The rows returned by the latest operation.
			 */
			static std::vector<std::map<std::string, std::string>> rows_;

			/**
			 * The callback to use for database operations.
			 * @param context The context.
			 * @param columnCount The amount of columns involved in the operation.
			 * @param columnValues The values of the columns.
			 * @param columnNames The names of the columns.
			 * @return The status code.
			 */
			static int callback(void* context, int columnCount, char** columnValues, char** columnNames);

			/**
			 * The ID of the Entity in the database.
			 */
			unsigned long id_;

		protected:
			/**
			 * Called when data is saved to the database.
			 * Used to store the object's fields.
			 */
			virtual void onSave();

			/**
			 * Sets the ID of the Entity.
			 * @param id The ID.
			 */
			void setID(const unsigned long& id);

		public:
			/**
			 * Initializes the database.
			 * @param databaseFile The file to load.
			 */
			static void initialize(const std::string& databaseFile);

			/**
			 * Executes the SQL statement.
			 * @param statement The statement to execute.
			 * @return The rows that were returned.
			 */
			static std::vector<std::map<std::string, std::string>> executeSQL(const std::string& statement, const std::string& file, const int& line);

			/**
			 * Adds a column to a database table.
			 * @param table The table.
			 * @param column The column name.
			 * @param attributes The attributes.
			 */
			static void addColumn(const std::string& table, const std::string& column, const std::string& attributes);

			/**
			 * Creates a database table.
			 * @param table The name of the table.
			 * @param columnsWithAttributes The columns and their attributes.
			 */
			static void createTable(const std::string& table, const std::map<std::string, std::string>& columnsWithAttributes);

			/**
			 * Inserts a bunch of rows into a database table.
			 * @param table The table.
			 * @param rowColumnValues The values of the rows.
			 */
			static void insert(const std::string& table, const std::vector<std::map<std::string, std::string>>& rowColumnValues);

			/**
			 * Inserts a row into a database table.
			 * @param table The table.
			 * @param columnValues The values of the columns.
			 * @return The ID of the row that was inserted.
			 */
			static unsigned long insert(const std::string& table, const std::map<std::string, std::string>& columnValues);

			/**
			 * Selects some data from the database.
			 * @param table The table.
			 * @param columns The columns to select.
			 * @param conditions The conditions to select on.
			 */
			static void select(const std::string& table, const std::vector<std::string>& columns, const std::string& conditions);

			/**
			 * Creates a new index on the table.
			 * @param table The table.
			 * @param index The name of the index.
			 * @param columns The columns on which to create the index.
			 * @param unique Whether the indices are unique.
			 */
			static void createIndex(const std::string& table, const std::string& index, const std::vector<std::string>& columns, const bool& unique = false);

			/**
			 * Filters some text for SQL input.
			 * @param sql The text to filter.
			 * @return The filtered text.
			 */
			static std::string filterSQL(std::string sql);

			/**
			 * Creates a new Entity.
			 */
			Entity() = default;

			/**
			 * Gets the ID of the Entity.
			 * @return The ID.
			 */
			unsigned long getID() const;

			/**
			 * Saves the Entity to the database.
			 */
			void save();
		};
	}
}
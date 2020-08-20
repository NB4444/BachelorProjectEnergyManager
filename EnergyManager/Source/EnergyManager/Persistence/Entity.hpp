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

#define ENERGY_MANAGER_PERSISTENCE_ENTITY_EXECUTE_SQL(STATEMENT) EnergyManager::Persistence::Entity::executeSQL(STATEMENT, __FILE__, __LINE__)

namespace EnergyManager {
	namespace Persistence {
		/**
		 * A persistent object.
		 */
		class Entity {
			/**
			 * The database to use.
			 */
			static sqlite3* database_;

			static std::vector<std::map<std::string, std::string>> rows_;

			static int callback(void* context, int columnCount, char** columnValues, char** columnNames);

		protected:
			/**
			 * Called when data is saved to the database.
			 * Used to store the object's fields.
			 */
			virtual void onSave();

		public:
			static void setDatabaseFile(const std::string& databaseFile);

			/**
			 * Executes the SQL statement.
			 * @param statement The statement to execute.
			 * @return The rows that were returned.
			 */
			static std::vector<std::map<std::string, std::string>> executeSQL(const std::string& statement, const std::string& file, const int& line);

			/**
			 * Creates a new Entity.
			 */
			Entity() = default;

			//~Entity() {
			//	// Close the connection
			//	//sqlite3_close(database_);
			//}

			void addColumn(const std::string& table, const std::string& column, const std::string& attributes);

			void createTable(const std::string& table, const std::map<std::string, std::string>& columnsWithAttributes);

			void insert(const std::string& table, std::vector<std::map<std::string, std::string>>& rowColumnValues);

			void insert(const std::string& table, const std::map<std::string, std::string>& columnValues);

			void select(const std::string& table, const std::vector<std::string>& columns, const std::string& conditions);

			/**
			 * Saves the Entity to the database.
			 */
			void save();
		};
	}
}
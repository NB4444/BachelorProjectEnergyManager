#pragma once

#include <functional>
#include <map>
#include <sqlite3.h>
#include <stdexcept>
#include <string>
#include <utility>

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
		sqlite3* database;

		/**
		 * The database ID of the Entity.
		 */
		int id_;

		/**
		 * The name of the Entity.
		 */
		std::string name_;

		/**
		 * Whether to save the Entity when its destructor is called.
		 */
		bool saveOnDestruct_;

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
			if(sqlite3_exec(database, statement.c_str(), callback.target<int(void*, int, char**, char**)>(), nullptr, &errorMessage)) {
				throw std::runtime_error("Could not execute SQL statement: " + std::string(errorMessage));
			}
		}

	protected:
		/**
		 * Called when data is saved to the database.
		 * Used to store the object's fields.
		 * @return The database row that should be saved.
		 */
		virtual std::map<std::string, std::string> onSave() = 0;

	public:
		static Type load(const int& id) {
			// TODO: Retrieve row from database as map of strings to strings and pass it to the load function below
			return Type(std::map<std::string, std::string> {});
		}

		/**
		 * Creates a new Entity.
		 * @param name The name of the Entity.
		 * @param saveOnDestruct Whether to save the Entity when its destructor is called.
		 */
		Entity(std::string name, const bool& saveOnDestruct = true)
			: name_(std::move(name))
			, saveOnDestruct_(saveOnDestruct) {
			// Open a new database connection
			if(sqlite3_open("database.sqlite", &database)) {
				throw std::runtime_error("Cannot open database");
			}
		}

		/**
		 * Destructs the Entity.
		 */
		~Entity() {
			if(saveOnDestruct_) {
				save();
			}

			sqlite3_close(database);
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
			// Create a table for the current Entity if it does not yet exist
			executeSQL("CREATE TABLE IF NOT EXISTS " + name_ + "(id INT PRIMARY KEY NOT NULL);");
			// TODO
		}
	};
}
# Find SQLite
find_package(SQLite3 REQUIRED)

# Find the include directory
set(SQLITE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIRS}")
message(STATUS "SQLite include directories: ${SQLITE_INCLUDE_DIRECTORIES}")

# Find the library
set(SQLITE_LIBRARIES "${SQLite3_LIBRARIES}")
message(STATUS "SQLite libraries: ${SQLITE_LIBRARIES}")
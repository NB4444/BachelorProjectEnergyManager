# Find the resources directory
set(PROJECT_RESOURCES_DIRECTORY "$ENV{HOME}/EnergyManager-Resources")
add_compile_definitions(PROJECT_RESOURCES_DIRECTORY="${PROJECT_RESOURCES_DIRECTORY}")
message(STATUS "Project resources directory: ${PROJECT_RESOURCES_DIRECTORY}")

# Find the database
set(PROJECT_DATABASE "${PROJECT_RESOURCES_DIRECTORY}/Database.sqlite")
add_compile_definitions(PROJECT_DATABASE="${PROJECT_DATABASE}")
message(STATUS "Project database: ${PROJECT_DATABASE}")

# Find the reporter
set(REPORTER_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/Reporter/lib${PROJECT_NAME}-Reporter.so")
add_compile_definitions(REPORTER_LIBRARY="${REPORTER_LIBRARY}")
message(STATUS "Reporter library: ${REPORTER_LIBRARY}")

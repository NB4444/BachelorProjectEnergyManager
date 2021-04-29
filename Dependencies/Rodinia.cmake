# Find the root directory
set(RODINIA_DIRECTORY "/applications/rodinia")
add_compile_definitions(RODINIA_DIRECTORY="${RODINIA_DIRECTORY}")
message(STATUS "Rodinia directory: ${RODINIA_DIRECTORY}")

# Find the data directory
set(RODINIA_DATA_DIRECTORY "${RODINIA_DIRECTORY}/data")
add_compile_definitions(RODINIA_DATA_DIRECTORY="${RODINIA_DATA_DIRECTORY}")
message(STATUS "Rodinia data directory: ${RODINIA_DATA_DIRECTORY}")
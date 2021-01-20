find_program(CONDA conda PATHS "~/anaconda3/bin")
message(STATUS "Conda: ${CONDA}")

# Add the environment
add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment"
        COMMAND "/bin/bash" "${CMAKE_CURRENT_SOURCE_DIR}/Build.sh" "${CONDA}" "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment" "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Set up the target
add_custom_target(
        ${PROJECT_NAME}-Visualizer
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/CondaEnvironment"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source"
)
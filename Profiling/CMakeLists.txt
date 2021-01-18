# Set up target
set(TARGET_NAME "${PROJECT_NAME}-Profiling")

# Get sources
set(SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source")
file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIRECTORY}/*.cpp" "${SOURCE_DIRECTORY}/*.cu")
message(STATUS "${TARGET_NAME} source files: ${SOURCE_FILES}")

# Add and configure target
add_executable(${TARGET_NAME} ${SOURCE_FILES})
set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(
        ${TARGET_NAME}
        PUBLIC "Source"
        PUBLIC "../Library/Source"
)
target_link_libraries(
        ${TARGET_NAME}
        PUBLIC "${PROJECT_NAME}-Library"
)
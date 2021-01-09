# Set up target
set(TARGET_NAME "${PROJECT_NAME}-Reporter")

# Get sources
set(SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source")
file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIRECTORY}/*.cpp" "${SOURCE_DIRECTORY}/*.cu")
message(STATUS "${TARGET_NAME} source files: ${SOURCE_FILES}")

# Add and configure target
add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(
        ${TARGET_NAME}
        PUBLIC Source
        #        PUBLIC ../Library/Source
        PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        PUBLIC "${CUDA_CUPTI_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        ${TARGET_NAME}
        #        ${PROJECT_NAME}-Library
        "cuda"
        "${CUDA_CUPTI_LIBRARY}"
)
# Set up directories
set(PROJECT_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/Source")

# Get sources
file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIRECTORY}/*.cpp" "${PROJECT_SOURCE_DIRECTORY}/*.cu")
message(STATUS "Source files: ${SOURCE_FILES}")

# Add and configure target
set(TARGET_NAME "${PROJECT_NAME}-Profiling-ActivityTrace")
add_executable(${TARGET_NAME} ${SOURCE_FILES})
set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(
        ${TARGET_NAME}
        PUBLIC Source
        PUBLIC ../Library/Source
        #        PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        #        PUBLIC "${CUDA_CUPTI_INCLUDE_DIRECTORY}"
)
target_link_libraries(
        ${TARGET_NAME}
        ${PROJECT_NAME}-Library
        #        "Threads::Threads"
        #        "cuda"
        #        "cudart"
        #        "${CUDA_CUPTI_LIBRARY}"
)
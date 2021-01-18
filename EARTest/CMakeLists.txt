cmake_minimum_required(VERSION 3.14.5)

# Set up the project
project(EnergyManager LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set up target
set(TARGET_NAME "${PROJECT_NAME}-EARTest")

# Enable EAR
set(EAR_ENABLED TRUE)
add_compile_definitions(EAR_ENABLED="${EAR_ENABLED}")

# Find the root directory
set(EAR_DIRECTORY "$ENV{EAR_INSTALL_PATH}")
message(STATUS "EAR directory: ${EAR_DIRECTORY}")

# Find the include directory
set(EAR_INCLUDE_DIRECTORIES "${EAR_DIRECTORY}/include")
message(STATUS "EAR include directories: ${EAR_INCLUDE_DIRECTORIES}")

# Find the library directory
set(EAR_LIBRARY_DIRECTORIES "${EAR_DIRECTORY}/lib")
add_compile_definitions(EAR_LIBRARY_DIRECTORIES="${EAR_LIBRARY_DIRECTORIES}")
message(STATUS "EAR library directories: ${EAR_LIBRARY_DIRECTORIES}")

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
        PUBLIC "${EAR_INCLUDE_DIRECTORIES}"
)
target_link_directories(
        ${TARGET_NAME}
        PUBLIC "${EAR_LIBRARY_DIRECTORIES}"
)
target_link_libraries(
        ${TARGET_NAME}
        PUBLIC "ear_api"
)
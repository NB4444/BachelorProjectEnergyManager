cmake_minimum_required(VERSION 3.14.5)

# Set up the project
project(EnergyManager LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set up target
set(TARGET_NAME "${PROJECT_NAME}-EARTest")

# Configure EAR
set(EAR_ENABLED TRUE)
add_compile_definitions(EAR_ENABLED="${EAR_ENABLED}")
set(EAR_DIRECTORY "/hpc/base/ctt/packages/EAR/ear")
set(EAR_INCLUDE_DIRECTORIES "${EAR_DIRECTORY}/include")
set(EAR_LIBRARY_DIRECTORIES "${EAR_DIRECTORY}/lib")
add_compile_definitions(EAR_LIBRARY_DIRECTORIES="${EAR_LIBRARY_DIRECTORIES}")
find_library(EAR_LIBRARIES ear.seq PATHS "${EAR_LIBRARY_DIRECTORIES}")
find_library(EAR_DAEMON_LIBRARIES earld "${EAR_LIBRARY_DIRECTORIES}")
find_library(EAR_API_LIBRARIES ear_api "${EAR_LIBRARY_DIRECTORIES}")
set(EAR_EACCT "${EAR_DIRECTORY}/bin/eacct")
add_compile_definitions(EAR_LIBRARIES="${EAR_LIBRARIES}")
add_compile_definitions(EAR_DAEMON_LIBRARIES="${EAR_DAEMON_LIBRARIES}")
add_compile_definitions(EAR_EACCT="${EAR_EACCT}")
message(STATUS "EAR directory: ${EAR_DIRECTORY}")
message(STATUS "EAR include directory: ${EAR_INCLUDE_DIRECTORIES}")
message(STATUS "EAR library directory: ${EAR_LIBRARY_DIRECTORIES}")
message(STATUS "EAR library: ${EAR_LIBRARIES}")
message(STATUS "EAR library daemon: ${EAR_DAEMON_LIBRARIES}")
message(STATUS "EAR library API: ${EAR_API_LIBRARIES}")
message(STATUS "EAR eacct: ${EAR_EACCT}")

# Get sources
set(SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source")
file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIRECTORY}/*.cpp" "${SOURCE_DIRECTORY}/*.cu")
message(STATUS "${TARGET_NAME} source files: ${SOURCE_FILES}")

# Add and configure target
add_library(${TARGET_NAME} STATIC ${SOURCE_FILES})
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
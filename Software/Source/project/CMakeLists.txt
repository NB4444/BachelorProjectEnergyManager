# Define the CUDA compiler to use
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Set up the project
cmake_minimum_required(VERSION 3.16)
project(Project LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
include_directories("${PROJECT_SOURCE_DIR}/../cuda_monitor/src")
set(ENV{LD_PRELOAD} "${PROJECT_SOURCE_DIR}/../cuda_monitor/src")
add_compile_definitions(CUDA_ENERGY_MONITOR_EXECUTABLE="${PROJECT_SOURCE_DIR}/../cuda_monitor/src/monitor")

# Get sources
file(GLOB_RECURSE SOURCE_FILES Source/*.cpp Source/*.cu)
message(STATUS "Source files: ${SOURCE_FILES}")

# Add and configure target
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
set_target_properties(
	${PROJECT_NAME}
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(
	${PROJECT_NAME}
	"${PROJECT_SOURCE_DIR}/../cuda_monitor/src/libmonitor.so"
	pthread
)
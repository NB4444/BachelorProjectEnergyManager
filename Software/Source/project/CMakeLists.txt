# Define the CUDA compiler to use
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_C_COMPILER /usr/bin/gcc-8)
set(CMAKE_CXX_COMPILER /usr/bin/g++-8)

# Set up the project
cmake_minimum_required(VERSION 2.8.12.2)
project(Project LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(SQLite3 REQUIRED)
add_compile_definitions(TEST_RESULTS_DATABASE="${PROJECT_SOURCE_DIR}/TestResults.sqlite")

# Set up the CUDA energy monitor
include_directories("${PROJECT_SOURCE_DIR}/../cuda_monitor/src")
add_compile_definitions(CUDA_ENERGY_MONITOR_DIRECTORY="${PROJECT_SOURCE_DIR}/../cuda_monitor/src")
#set(ENV{LD_PRELOAD} ${CUDA_ENERGY_MONITOR_LIBRARY})

# Get sources
file(GLOB_RECURSE SOURCE_FILES Source/*.cpp Source/*.cu)
message(STATUS "Source files: ${SOURCE_FILES}")

# Add and configure target
add_executable(${PROJECT_NAME} ${SOURCE_FILES} Source/Hardware/Device.cpp Source/Hardware/Device.hpp Source/Hardware/GPU.cpp Source/Hardware/GPU.hpp)
set_target_properties(
	${PROJECT_NAME}
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(
	${PROJECT_NAME}
	PUBLIC Source
	PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	PUBLIC ${SQLite3_INCLUDE_DIRS}
)
target_link_libraries(
	${PROJECT_NAME}
	"${PROJECT_SOURCE_DIR}/../cuda_monitor/src/libmonitor.so"
	"${CMAKE_CUDA_RUNTIME_LIBRARY}"
	cuda
	/usr/local/cuda-10.1/lib64/libcudart.so
	cupti
	pthread
	${SQLite3_LIBRARIES}
)
cmake_minimum_required(VERSION 3.16.3)

# Configure CUDA compiler
set(CMAKE_CUDA_COMPILER nvcc)
if(NOT EXISTS ${CMAKE_CUDA_COMPILER})
	set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

# Set up the project
project(EnergyManager LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure pthreads
find_package(Threads REQUIRED)
message(STATUS "Threads libraries: ${CMAKE_THREAD_LIBS_INIT}")

# Configure CUDA
find_package(CUDA 10.1 REQUIRED)
message(STATUS "CUDA include directories: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Configure SQLite
find_package(SQLite3 REQUIRED)
message(STATUS "SQLite3 include directories: ${SQLite3_INCLUDE_DIRS}")
message(STATUS "SQLite3 libraries: ${SQLite3_LIBRARIES}")

## Set up the CUDA energy monitor
#include_directories("${PROJECT_SOURCE_DIR}/../cuda_monitor/src")
#add_compile_definitions(CUDA_ENERGY_MONITOR_DIRECTORY="${PROJECT_SOURCE_DIR}/../cuda_monitor/src")
##set(ENV{LD_PRELOAD} ${CUDA_ENERGY_MONITOR_LIBRARY})

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
target_include_directories(
	${PROJECT_NAME}
	PUBLIC Source
	PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	PUBLIC /usr/local/cuda/extras/CUPTI/include
	PUBLIC /cm/shared/apps/cuda10.1/toolkit/10.1.243/extras/CUPTI/include
	PUBLIC ${SQLite3_INCLUDE_DIRS}
)
target_link_libraries(
	${PROJECT_NAME}
	${CMAKE_THREAD_LIBS_INIT}
	cuda
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../../../extras/CUPTI/lib64/libcupti.so"
	nvidia-ml
	${SQLite3_LIBRARIES}
)
# Define the CUDA compiler to use
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Set up the project
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_compile_definitions(TEST_RESULTS_DATABASE="${PROJECT_SOURCE_DIR}/TestResults.sqlite")

# Set up the CUDA energy monitor
include_directories("${PROJECT_SOURCE_DIR}/cuda_monitor/src")
add_compile_definitions(CUDA_ENERGY_MONITOR_DIRECTORY="${PROJECT_SOURCE_DIR}/cuda_monitor/src")
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
	PUBLIC /usr/local/cuda/extras/CUPTI/include
	PUBLIC /cm/shared/apps/cuda10.1/toolkit/10.1.243/extras/CUPTI/include
	PUBLIC "${PROJECT_SOURCE_DIR}/sqlite"
)
target_link_libraries(
	${PROJECT_NAME}
	cuda
	${CMAKE_CUDA_RUNTIME_LIBRARY}
	${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
	cupti
	pthread
	"${PROJECT_SOURCE_DIR}/sqlite/.libs/libsqlite3.so"
)
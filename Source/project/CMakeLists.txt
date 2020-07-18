cmake_minimum_required(VERSION 3.16)
project(Project CUDA)

set(CMAKE_CUDA_STANDARD 14)

# Get sources
file(GLOB_RECURSE SOURCE_FILES Source/*.cpp Source/*.cu)
message(STATUS "Source files: ${SOURCE_FILES}")

# Add target
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set_target_properties(
	${PROJECT_NAME}
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)
# Set up target
set(TARGET_NAME "${PROJECT_NAME}-ShowFrequencies")

# Get sources
set(SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source")
file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIRECTORY}/*.cpp" "${SOURCE_DIRECTORY}/*.cu")

# Find the root directory
set(EAR_DIRECTORY "$ENV{EAR_INSTALL_PATH}")

# Find the include directory
set(EAR_INCLUDE_DIRECTORIES "${EAR_DIRECTORY}/include")

# Find the library directory
set(EAR_LIBRARY_DIRECTORIES "${EAR_DIRECTORY}/lib")
add_compile_definitions(EAR_LIBRARY_DIRECTORIES="${EAR_LIBRARY_DIRECTORIES}")

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
	PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
	PUBLIC "${EAR_INCLUDE_DIRECTORIES}"
)
target_link_directories(
	${TARGET_NAME}
	PUBLIC "${EAR_LIBRARY_DIRECTORIES}"
)
target_link_libraries(
	${TARGET_NAME}
	PUBLIC "cuda"
	#        PUBLIC "ear_api_debug"
	PUBLIC "ear_api"
	PUBLIC "irc"
	PUBLIC "nvidia-ml"
)
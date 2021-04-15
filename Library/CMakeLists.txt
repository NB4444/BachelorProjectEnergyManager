# Set up target
set(TARGET_NAME "${PROJECT_NAME}-Library")

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
	PUBLIC "${BOOST_INCLUDE_DIRECTORIES}"
	PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
	PUBLIC "${CUDA_CUPTI_INCLUDE_DIRECTORIES}"
	PUBLIC "${EAR_INCLUDE_DIRECTORIES}"
	PUBLIC "${SQLITE_INCLUDE_DIRECTORIES}"
)
target_link_libraries(
	${TARGET_NAME}
	PUBLIC "${CUDA_CUPTI_LIBRARIES}"
	PUBLIC "${SQLITE_LIBRARIES}"
	PUBLIC "Boost::filesystem"
	PUBLIC "Boost::math_c99"
	PUBLIC "Boost::program_options"
	PUBLIC "Boost::serialization"
	PUBLIC "Boost::stacktrace_addr2line"
	PUBLIC "Boost::unit_test_framework"
	PUBLIC "Threads::Threads"
	PUBLIC "cuda"
	PUBLIC "dl"
#	PUBLIC "irc"
	PUBLIC "nvidia-ml"
)
if (DEFINED EAR_ENABLED)
	message(STATUS "Adding ear libraries to ${TARGET_NAME}...")
	target_link_directories(
		${TARGET_NAME}
		PUBLIC "${EAR_LIBRARY_DIRECTORIES}"
	)
	target_link_libraries(
		${TARGET_NAME}
		PUBLIC "ear_api"
	)
endif ()
# Find Boost
find_package(
	Boost 1.65 REQUIRED
	COMPONENTS
	filesystem
	math_c99
	program_options
	serialization
	stacktrace_addr2line
	unit_test_framework
)

# Find the include directory
set(BOOST_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIR}")
message(STATUS "Boost include directories: ${BOOST_INCLUDE_DIRECTORIES}")
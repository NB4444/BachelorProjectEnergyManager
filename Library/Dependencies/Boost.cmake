# Find Boost
#set(Boost_NO_BOOST_CMAKE "on")
find_package(Boost 1.70 REQUIRED COMPONENTS math_c99 program_options unit_test_framework serialization filesystem)

# Find the include directory
set(BOOST_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
message(STATUS "Boost include directories: ${BOOST_INCLUDE_DIRECTORIES}")

# Find the library
set(BOOST_LIBRARIES "${Boost_LIBRARIES}")
message(STATUS "Boost libraries: ${BOOST_LIBRARIES}")
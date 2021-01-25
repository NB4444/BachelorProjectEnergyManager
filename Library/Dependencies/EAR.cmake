# Find the root directory
set(EAR_DIRECTORY "$ENV{EAR_INSTALL_PATH}")
message(STATUS "EAR directory: ${EAR_DIRECTORY}")

# Find the include directory
set(EAR_INCLUDE_DIRECTORIES "${EAR_DIRECTORY}/include")
message(STATUS "EAR include directories: ${EAR_INCLUDE_DIRECTORIES}")

# Find the library directory
set(EAR_LIBRARY_DIRECTORIES "${EAR_DIRECTORY}/lib")
add_compile_definitions(EAR_LIBRARY_DIRECTORIES="${EAR_LIBRARY_DIRECTORIES}")
message(STATUS "EAR library directories: ${EAR_LIBRARY_DIRECTORIES}")

# Find the library
find_library(EAR_LIBRARIES ear.seq PATHS "${EAR_LIBRARY_DIRECTORIES}")
add_compile_definitions(EAR_LIBRARIES="${EAR_LIBRARIES}")
message(STATUS "EAR libraries: ${EAR_LIBRARIES}")

# Find the daemon library
find_library(EAR_DAEMON_LIBRARIES earld "${EAR_LIBRARY_DIRECTORIES}")
add_compile_definitions(EAR_DAEMON_LIBRARIES="${EAR_DAEMON_LIBRARIES}")
message(STATUS "EAR daemon libraries: ${EAR_DAEMON_LIBRARIES}")

# Find eacct
set(EAR_EACCT "${EAR_DIRECTORY}/bin/eacct")
add_compile_definitions(EAR_EACCT="${EAR_EACCT}")
message(STATUS "EAR eacct: ${EAR_EACCT}")

# Enable EAR if available
if (EXISTS ${EAR_DIRECTORY})
    message(STATUS "EAR found, enabling EAR functionality")
    set(EAR_ENABLED TRUE)
    add_compile_definitions(EAR_ENABLED="${EAR_ENABLED}")
else ()
    message(STATUS "EAR not found, disabling EAR functionality")
endif ()
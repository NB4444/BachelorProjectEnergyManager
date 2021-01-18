# TODO: Add necessary licenses for all libraries

# Set up dependencies
include(Dependencies/Boost.cmake)
include(Dependencies/CUDA.cmake)
include(Dependencies/EAR.cmake)
include(Dependencies/MPI.cmake)
include(Dependencies/NVIDIACodeSamples.cmake)
include(Dependencies/Rodinia.cmake)
include(Dependencies/SLURM.cmake)
include(Dependencies/SQLite.cmake)
include(Dependencies/Threads.cmake)

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
        PUBLIC "${SQLITE_INCLUDE_DIRECTORY}"
        PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        PUBLIC "${CUDA_CUPTI_INCLUDE_DIRECTORIES}"
        PUBLIC "${BOOST_INCLUDE_DIRECTORIES}"
        # FIXME: Quick fix to fix the wrong boost includes from being used
        #        PUBLIC "/hpc/base/ctt/packages/boost/1.70.0/intel/include"
        PUBLIC "${EAR_INCLUDE_DIRECTORIES}"
)
target_link_libraries(
        ${TARGET_NAME}
        PUBLIC "Threads::Threads"
        PUBLIC "${SQLITE_LIBRARIES}"
        PUBLIC "cuda"
        PUBLIC "${CUDA_CUPTI_LIBRARIES}"
        PUBLIC "nvidia-ml"
        PUBLIC "dl"
        PUBLIC "${BOOST_LIBRARIES}"
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
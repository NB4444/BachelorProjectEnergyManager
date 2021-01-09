# TODO: Add necessary licenses for all libraries
# TODO: Add ensmallen citation from here https://github.com/mlpack/ensmallen

# Set up target
set(TARGET_NAME "${PROJECT_NAME}-Library")

# Set up dependencies
include(Dependencies.cmake)

# Get sources
set(SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Source")
file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIRECTORY}/*.cpp" "${SOURCE_DIRECTORY}/*.cu")
message(STATUS "${TARGET_NAME} source files: ${SOURCE_FILES}")

# Add and configure target
add_library(${TARGET_NAME} STATIC ${SOURCE_FILES})
#set_property(TARGET ${PROJECT_NAME}-Library PROPERTY POSITION_INDEPENDENT_CODE OFF)
add_dependencies(
        ${TARGET_NAME}
        ensmallen
        mlpack
)
set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(
        ${TARGET_NAME}
        PUBLIC "Source"
        #        PUBLIC "${MPI_INCLUDE_DIRECTORY}"
        PUBLIC "${SQLITE_INCLUDE_DIRECTORY}"
        PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        PUBLIC "${CUDA_CUPTI_INCLUDE_DIRECTORY}"
        #        PUBLIC "${BOOST_INCLUDE_DIRECTORY}"
        # FIXME: Quick fix to fix the wrong boost includes from being used
        PUBLIC "/hpc/base/ctt/packages/boost/1.70.0/intel/include"
        PUBLIC "${OPENBLAS_INCLUDE_DIRECTORY}"
        PUBLIC "${ARMADILLO_INCLUDE_DIRECTORY}"
        PUBLIC "${ENSMALLEN_INCLUDE_DIRECTORY}"
        PUBLIC "${MLPACK_INCLUDE_DIRECTORY}"
)
#target_link_libraries(
#        ${TARGET_NAME}
#        "Threads::Threads"
#        imf
#        #        "${MPI_LIBRARY}"
#        "${SQLITE_LIBRARY}"
#        "${CUDA_LIBRARY}"
#        "${CUDA_RUNTIME_LIBRARY}"
#        #        "${CUDA_STATIC_RUNTIME_LIBRARY}"
#        "${CUDA_CUPTI_LIBRARY}"
#        #        "${CUDA_NVML_LIBRARY}"
#        "nvidia-ml"
#        #        "OpenMP::OpenMP_CXX"
#        "${BOOST_LIBRARY}"
#        "${OPENBLAS_LIBRARY}"
#        "${ARMADILLO_LIBRARY}"
#        "${MLPACK_LIBRARY}"
#)
target_link_libraries(
        ${TARGET_NAME}
        "Threads::Threads"
        "${SQLITE_LIBRARY}"
        "cuda"
        "${CUDA_CUPTI_LIBRARY}"
        "nvidia-ml"
        "dl"
        "${BOOST_LIBRARY}"
        "${OPENBLAS_LIBRARY}"
        "${ARMADILLO_LIBRARY}"
        "${MLPACK_LIBRARY}"
)

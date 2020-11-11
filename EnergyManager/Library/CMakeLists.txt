# Set up directories
set(PROJECT_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/Source")

# Configure pthreads
find_package(Threads REQUIRED)
message(STATUS "Threads libraries: ${CMAKE_THREAD_LIBS_INIT}")

# Configure CUDA
find_package(CUDA 10.1 REQUIRED)

# Configure SQLite
find_package(SQLite3 REQUIRED)
message(STATUS "SQLite3 include directories: ${SQLite3_INCLUDE_DIRS}")
message(STATUS "SQLite3 libraries: ${SQLite3_LIBRARIES}")

# Configure OpenMP for mlpack
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	message(STATUS "OpenMP C flags: ${OpenMP_C_FLAGS}")
	message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
	message(STATUS "OpenMP EXE linker flags: ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Configure Boost for mlpack
find_package(Boost REQUIRED COMPONENTS serialization)

# Configure armadillo for ensmallen
find_path(
	ARMADILLO_INCLUDE_DIRECTORY
	NAMES arma_version.hpp
	PATHS "$ENV{ProgramFiles}/armadillo_bits/" /usr/local/include/armadillo_bits/ /usr/include/armadillo_bits/ ~/applications/armadillo/include/armadillo_bits
)
message(STATUS "armadillo include directories: ${ARMADILLO_INCLUDE_DIRECTORY}")
find_library(
	ARMADILLO_LIBRARY
	NAMES armadillo
	PATHS "$ENV{ProgramFiles}/armadillo_bits/" /usr/lib64/ /usr/lib/ /usr/local/lib64/ /usr/local/ ~/applications/armadillo/lib64
)
message(STATUS "armadillo library: ${ARMADILLO_LIBRARY}")

# Configure mlpack
find_path(
	MLPACK_INCLUDE_DIRECTORY
	NAMES core.hpp prereqs.hpp
	PATHS "$ENV{ProgramFiles}/mlpack/" /usr/local/include/mlpack/ /usr/include/mlpack/ ~/applications/mlpack/include
)
message(STATUS "mlpack include directories: ${MLPACK_INCLUDE_DIRECTORY}")
find_library(
	MLPACK_LIBRARY
	NAMES mlpack
	PATHS "$ENV{ProgramFiles}/mlpack/" /usr/lib64/ /usr/lib/ /usr/local/lib64/ /usr/local/ ~/applications/mlpack/
)
message(STATUS "mlpack library: ${MLPACK_LIBRARY}")

# Configure ensmallen
find_path(
	ENSMALLEN_INCLUDE_DIRECTORY
	NAMES ensmallen.hpp
	PATHS "$ENV{ProgramFiles}/ensmallen_bits/" /usr/local/include/ /usr/include/ ~/applications/ensmallen/include
)
message(STATUS "ensmallen include directories: ${ENSMALLEN_INCLUDE_DIRECTORY}")

# Get sources
file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIRECTORY}/*.cpp" "${PROJECT_SOURCE_DIRECTORY}/*.cu")
message(STATUS "Source files: ${SOURCE_FILES}")

# Add and configure target
set(TARGET_NAME "${PROJECT_NAME}-Library")
add_library(${TARGET_NAME} STATIC ${SOURCE_FILES})
set_target_properties(
	${TARGET_NAME}
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(
	${TARGET_NAME}
	PUBLIC Source
	PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	PUBLIC /usr/local/cuda/extras/CUPTI/include
	PUBLIC /cm/shared/apps/cuda10.1/toolkit/10.1.243/extras/CUPTI/include
	PUBLIC ${SQLite3_INCLUDE_DIRS}
	PUBLIC ${ARMADILLO_INCLUDE_DIRECTORY}
	PUBLIC ${MLPACK_INCLUDE_DIRECTORY}
	PUBLIC ${ENSMALLEN_INCLUDE_DIRECTORY}
)
target_link_libraries(
	${TARGET_NAME}
	${CMAKE_THREAD_LIBS_INIT}
	cuda
	${CMAKE_CUDA_RUNTIME_LIBRARY}
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../../../lib64/libcudart.so"
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../../../extras/CUPTI/lib64/libcupti.so"
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../../../lib64/stubs/libnvidia-ml.so"
	nvidia-ml
	${SQLite3_LIBRARIES}
	OpenMP::OpenMP_CXX
	${Boost_LIBRARIES}
	${ARMADILLO_LIBRARY}
	${MLPACK_LIBRARY}
)
# TODO: Add necessary licenses for all libraries
# TODO: Add ensmallen citation from here https://github.com/mlpack/ensmallen

# Enable symbol generation for debug output
#target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-rdynamic>")

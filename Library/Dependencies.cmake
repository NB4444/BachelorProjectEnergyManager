include(ExternalProject)

# Configure MPI
#find_path(
#        MPI_INCLUDE_DIRECTORY
#        NAMES mpi.h
#        PATHS "/hpc/base/platform/mpi/8.1.1/include"
#)
#find_library(
#        MPI_LIBRARY
#        NAMES libmpi mpi
#        PATHS "/hpc/base/platform/mpi/8.1.1/lib/linux_amd64"
#)
find_package(MPI REQUIRED)
set(MPI_INCLUDE_DIRECTORY "${MPI_CXX_INCLUDE_DIRS}")
set(MPI_LIBRARY "${MPI_CXX_LIBRARIES}")
find_program(MPI_MPIEXEC mpiexec)
add_compile_definitions(MPI_MPIEXEC="${MPI_MPIEXEC}")
message(STATUS "MPI include directory: ${MPI_INCLUDE_DIRECTORY}")
message(STATUS "MPI library: ${MPI_LIBRARY}")
message(STATUS "MPI mpiexec: ${MPI_MPIEXEC}")

# Configure SQLite
find_package(SQLite3 REQUIRED)
set(SQLITE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIRS}")
set(SQLITE_LIBRARY "${SQLite3_LIBRARIES}")
message(STATUS "SQLite3 include directories: ${SQLITE_INCLUDE_DIRECTORY}")
message(STATUS "SQLite3 libraries: ${SQLITE_LIBRARY}")

## Configure OpenMP
#find_package(OpenMP REQUIRED)
#if (OPENMP_FOUND)
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#    message(STATUS "OpenMP C flags: ${OpenMP_C_FLAGS}")
#    message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
#    message(STATUS "OpenMP EXE linker flags: ${OpenMP_EXE_LINKER_FLAGS}")
#endif ()
#message(STATUS "OpenMP libraries: ${OpenMP_LIBRARIES}")

# Configure OpenBLAS
set(OPENBLAS_INCLUDE_DIRECTORY "${PROJECT_LIBRARY_INCLUDES_DIRECTORY}")
set(OPENBLAS_LIBRARY "${PROJECT_LIBRARY_FILES_DIRECTORY}/libblas.a")
message(STATUS "OpenBLAS include directory: ${OPENBLAS_INCLUDE_DIRECTORY}")
message(STATUS "OpenBLAS library: ${OPENBLAS_LIBRARY}")
ExternalProject_Add(
        OpenBLAS
        URL "https://github.com/xianyi/OpenBLAS/archive/v0.3.5.tar.gz"
        URL_MD5 "579bda57f68ea6e9074bf5780e8620bb"
        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'" "-DTARGET=HASWELL"
)

# Configure LAPACK
set(LAPACK_INCLUDE_DIRECTORY "${PROJECT_LIBRARY_INCLUDES_DIRECTORY}")
set(LAPACK_LIBRARY "${PROJECT_LIBRARY_FILES_DIRECTORY}/liblapack.a")
message(STATUS "LAPACK include directory: ${LAPACK_INCLUDE_DIRECTORY}")
message(STATUS "LAPACK library: ${LAPACK_LIBRARY}")
ExternalProject_Add(
        LAPACK
        URL "http://www.netlib.org/lapack/lapack-3.8.0.tar.gz"
        URL_MD5 "96591affdbf58c450d45c1daa540dbd2"
        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'"
)

## Configure ARPACK
#set(ARPACK_INCLUDE_DIRECTORY "${PROJECT_LIBRARIES_DIRECTORY}/include")
#set(ARPACK_LIBRARY "${PROJECT_LIBRARIES_DIRECTORY}/lib/libarpack.so")
#message(STATUS "ARPACK include directory: ${ARPACK_INCLUDE_DIRECTORY}")
#message(STATUS "ARPACK library: ${ARPACK_LIBRARY}")
#ExternalProject_Add(
#        ARPACK
#        URL "https://github.com/opencollab/arpack-ng/archive/3.6.3.tar.gz"
#        URL_MD5 "a159a998d1805e18a68ca1995f79a245"
#        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'"
#)

## Configure SuperLU
#set(SUPERLU_INCLUDE_DIRECTORY "${PROJECT_LIBRARIES_DIRECTORY}/include")
#set(SUPERLU_LIBRARY "${PROJECT_LIBRARIES_DIRECTORY}/lib/libsuperlu.so")
#message(STATUS "SuperLU include directory: ${SUPERLU_INCLUDE_DIRECTORY}")
#message(STATUS "SuperLU library: ${SUPERLU_LIBRARY}")
#ExternalProject_Add(
#        SuperLU
#        URL "https://github.com/xiaoyeli/superlu/archive/v5.2.2.tar.gz"
#        URL_MD5 "1e93259572bd2412674ed809a1446bd3"
#        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'"
#)

# Configure Armadillo
set(ARMADILLO_INCLUDE_DIRECTORY "${PROJECT_LIBRARY_INCLUDES_DIRECTORY}")
set(ARMADILLO_LIBRARY "${PROJECT_LIBRARY_FILES_DIRECTORY}/libarmadillo.so")
message(STATUS "Armadillo include directory: ${ARMADILLO_INCLUDE_DIRECTORY}")
message(STATUS "Armadillo library: ${ARMADILLO_LIBRARY}")
ExternalProject_Add(
        Armadillo
        URL "https://downloads.sourceforge.net/project/arma/armadillo-8.600.1.tar.xz"
        URL_MD5 "3380ccfc68d8eb318af0253b92301a66"
        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'" "-DDETECT_HDF5=false"
        DEPENDS OpenBLAS LAPACK
)

# Configure ensmallen
set(ENSMALLEN_INCLUDE_DIRECTORY "${PROJECT_LIBRARY_INCLUDES_DIRECTORY}")
message(STATUS "ensmallen include directory: ${ENSMALLEN_INCLUDE_DIRECTORY}")
ExternalProject_Add(
        ensmallen
        URL "https://github.com/mlpack/ensmallen/archive/2.15.1.tar.gz"
        URL_MD5 "74bb5cb4ee1525f5569c48848161b225"
        #        URL "https://github.com/mlpack/ensmallen/archive/ensmallen-1.14.0.tar.gz"
        #        URL_MD5 "3c4782ce15c0f6bc961df116dc8acf1f"
        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'" "-DCMAKE_CXX_STANDARD_LIBRARIES='-lgfortran -lpthread'"
        DEPENDS Armadillo
)

# Configure Boost
set(Boost_NO_BOOST_CMAKE "on")
find_package(Boost 1.67 REQUIRED COMPONENTS math_c99 program_options unit_test_framework serialization)
#set(BOOST_ROOT_DIRECTORY "${Boost_INCLUDE_DIR}")
#if(NOT BOOST_ROOT_DIRECTORY AND EXISTS "/hpc/base/ctt/packages/boost/1.70.0/intel")
#    set(BOOST_ROOT_DIRECTORY "/hpc/base/ctt/packages/boost/1.70.0/intel")
#endif()
set(BOOST_INCLUDE_DIRECTORY "${Boost_INCLUDE_DIRS}")
set(BOOST_LIBRARIES_DIRECTORY "${Boost_LIBRARY_DIRS}")
set(BOOST_LIBRARY "${Boost_LIBRARIES}")
#set(BOOST_INCLUDE_DIRECTORY "${BOOST_ROOT_DIRECTORY}/include")
#set(BOOST_ROOT_DIRECTORY "${PROJECT_LIBRARIES_DIRECTORY}/boost")
#set(BOOST_INCLUDE_DIRECTORY "${PROJECT_LIBRARIES_DIRECTORY}/include")
#set(BOOST_LIBRARY "${PROJECT_LIBRARIES_DIRECTORY}/lib/libboost.so")
#message(STATUS "Boost root directory: ${BOOST_ROOT_DIRECTORY}")
message(STATUS "Boost include directory: ${BOOST_INCLUDE_DIRECTORY}")
message(STATUS "Boost libraries directory: ${BOOST_LIBRARIES_DIRECTORY}")
message(STATUS "Boost library: ${BOOST_LIBRARY}")
#ExternalProject_Add(
#        Boost
#        URL "https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz"
#        URL_MD5 "b50944c0c13f81ce2c006802a1186f5a"
#        CONFIGURE_COMMAND "./bootstrap.sh" "--prefix='${BOOST_ROOT_DIRECTORY}'" "--with-libraries=program_options,test,serialization"
#        BUILD_COMMAND "./b2" "install"
#        BUILD_IN_SOURCE true
#        INSTALL_COMMAND ""
#        INSTALL_DIR "${BOOST_ROOT_DIRECTORY}"
#

# Configure mlpack
ExternalProject_Add(
        mlpack
        #        URL "https://github.com/mlpack/mlpack/archive/mlpack-3.0.4.tar.gz"
        #        URL_MD5 "361862a4d3f210ccb2181e5c1049f2b2"
        URL "https://github.com/mlpack/mlpack/archive/3.4.2.tar.gz"
        URL_MD5 "56b7056a50361089015fa8b2ce923cf8"
        #        URL "https://github.com/mlpack/mlpack/archive/3.2.1.tar.gz"
        #        URL_MD5 "1111f6ef7b41657ae2107f0182d2e157"
        #        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'" "-DBoost_NO_BOOST_CMAKE=on" "-DCMAKE_CXX_STANDARD_LIBRARIES='-lgfortran -lboost_program_options -lboost_unit_test_framework -lboost_serialization'"
        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'" "-DBOOST_INCLUDEDIR:PATH='${BOOST_INCLUDE_DIRECTORY}'" "-DBOOST_LIBRARYDIR:PATH='${BOOST_LIBRARIES_DIRECTORY}'" "-DBoost_NO_BOOST_CMAKE=on" "-DCMAKE_CXX_STANDARD_LIBRARIES='-lirc -lgfortran -lpthread -lboost_math_c99 -lboost_program_options -lboost_unit_test_framework -lboost_serialization'"
        #        CMAKE_ARGS "-DCMAKE_C_COMPILER:PATH='${CMAKE_C_COMPILER}'" "-DCMAKE_CXX_COMPILER:PATH='${CMAKE_CXX_COMPILER}'" "-DCMAKE_Fortran_COMPILER:PATH='${CMAKE_Fortran_COMPILER}'" "-DCMAKE_INSTALL_PREFIX:PATH='${PROJECT_LIBRARIES_DIRECTORY}'" "-DBOOST_ROOT='${BOOST_ROOT_DIRECTORY}'" "-DCMAKE_CXX_STANDARD_LIBRARIES='-lgfortran'"
        DEPENDS Armadillo ensmallen
)
set(MLPACK_INCLUDE_DIRECTORY "${PROJECT_LIBRARY_INCLUDES_DIRECTORY}")
set(MLPACK_LIBRARY "${PROJECT_LIBRARY_FILES_DIRECTORY}/libmlpack.so")
message(STATUS "mlpack include directory: ${MLPACK_INCLUDE_DIRECTORY}")
message(STATUS "mlpack library: ${MLPACK_LIBRARY}")

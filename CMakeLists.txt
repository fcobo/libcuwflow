# CMake to build libcuwflow and test executable

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cuwflow)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost REQUIRED)

set (REQUIRED_LIBRARIES ${OpenCV_LIBS} )

IF(Boost_FOUND)
   MESSAGE( STATUS "BOOST Libraries " ${Boost_LIBRARIES} )
   LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
   MESSAGE( STATUS "BOOST Include " ${Boost_INCLUDE_DIRS} )
   INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS})
ELSE()
   MESSAGE( WARNING "You must set BOOST_ROOT variable" )
ENDIF(Boost_FOUND)

INCLUDE(FindCUDA)

#------------------------------------------------
# DIRS
#------------------------------------------------
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tests)

#-----------------------------------------------

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_10,code=sm_10)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

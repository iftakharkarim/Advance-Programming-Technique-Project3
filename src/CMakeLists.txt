cmake_minimum_required(VERSION 3.8)

project(P3 LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS -pthread)

include_directories(cthreads cuda mpi)
include_directories(${MPI_INCLUDE_PATH})

add_library(P3LIB STATIC
		cthreads/complex.cc
		cthreads/complex.h
		cthreads/input_image.cc
		cthreads/input_image.h
		cuda/complex.cuh
		)

add_executable(P31 cthreads/MainThread.cc)
add_executable(P32 mpi/p32.cpp)
add_executable(P33 cuda/mainofCuda.cu)

set_target_properties(P33 PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
target_link_libraries(P33 PRIVATE P3LIB)
target_link_libraries(P32 ${MPI_LIBRARIES})
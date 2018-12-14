cmake_minimum_required(VERSION 3.8)

project(p3 LANGUAGES CXX CUDA)

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

add_executable(p31 cthreads/MainThread.cc)
add_executable(p32 mpi/p32.cpp)
add_executable(p33 cuda/mainofCuda.cu)

set_target_properties(p33 PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
target_link_libraries(p33 PRIVATE P3LIB)
target_link_libraries(p32 ${MPI_LIBRARIES})
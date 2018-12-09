cmake_minimum_required(VERSION 3.8)

project(P3 LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)

include_directories(src)

add_subdirectory(P31)
add_subdirectory(P32)
add_subdirectory(P33)

add_library(${PROJECT_NAME} SHARED
        src/complex.cc
        src/complex.h
        src/input_image.cc
        src/input_image.h
        )

add_library(P31 STATIC
        src/MainThread.cc)

add_library(P32 STATIC
        )

add_library(P33 STATIC
        )

target_compile_features(P31 PUBLIC cxx_std_11)
target_compile_features(P32 PUBLIC cxx_std_11)
target_compile_features(P33 PUBLIC cxx_std_11)

set_target_properties(P33 PROPERTIES CUDA_SEPERABALE_COMPILATION ON)

add_executable(P31 src/MainThread.cc)

#add_executable(P32 src/)

#add_executable(P33 src/)
cmake_minimum_required(VERSION 3.0.0)
project(MPI_exercise VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb -fno-omit-frame-pointer -fno-inline-functions")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

find_package(MPI REQUIRED)

add_executable(MPI_exercise
    main.cc
)
target_link_libraries(MPI_exercise PRIVATE
    MPI::MPI_CXX
)


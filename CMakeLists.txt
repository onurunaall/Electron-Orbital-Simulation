cmake_minimum_required(VERSION 3.24)
project(atoms_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile for the GPU in this machine unless told otherwise
# (e.g. cmake -DCMAKE_CUDA_ARCHITECTURES=86 for an RTX 30xx).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Code shared by both programs: particle sampling and the probability-current update.
add_library(atoms_common STATIC common/particles.cu)
target_include_directories(atoms_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Both programs pass __device__ lambdas to Thrust (copy_if, transform_reduce).
target_compile_options(atoms_common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(atom_realtime realtime/main.cu realtime/rasterizer.cu)
target_link_libraries(atom_realtime PRIVATE atoms_common)

add_executable(atom_raytracer raytracer/main.cu raytracer/bvh.cu raytracer/tracer.cu)
target_link_libraries(atom_raytracer PRIVATE atoms_common)

# CPU checks of the shared host+device math; needs the CUDA toolkit but no GPU.
add_executable(atoms_tests tests/cpu_tests.cu)
target_link_libraries(atoms_tests PRIVATE atoms_common)
enable_testing()
add_test(NAME cpu_tests COMMAND atoms_tests)

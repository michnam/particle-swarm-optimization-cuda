cmake_minimum_required(VERSION 3.19)
project(pso CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(SFML_DIR "C:/SFML/SFML-2.5.1/lib/cmake/SFML")
find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)

add_executable(pso main.cu pso.cu pso.cuh)
target_link_libraries(pso sfml-graphics sfml-audio)
set_target_properties(
        pso
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
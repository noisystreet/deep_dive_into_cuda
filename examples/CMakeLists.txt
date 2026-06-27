cmake_minimum_required(VERSION 3.18)
project(cuda_examples LANGUAGES CUDA CXX)

# Enable verbose output for nvcc (shows full command lines in build log)
# --keep preserves all intermediate compilation artifacts for analysis
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --verbose --keep")

# Optional: also show full compile/link command lines in Makefile output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set default CUDA architecture for all targets
set(CMAKE_CUDA_ARCHITECTURES "89")

add_executable(vector_add vector_add.cu)
add_executable(wmma_matmul wmma_matmul.cu)
add_executable(graph_demo graph_demo.cu)
add_executable(streams_demo streams_demo.cu)

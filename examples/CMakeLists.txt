cmake_minimum_required(VERSION 3.18)
project(vector_add LANGUAGES CUDA CXX)

# Enable verbose output for nvcc (shows full command lines in build log)
# --keep preserves all intermediate compilation artifacts for analysis
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --verbose --keep")

# Optional: also show full compile/link command lines in Makefile output
set(CMAKE_VERBOSE_MAKEFILE ON)

add_executable(vector_add vector_add.cu)

# Set default CUDA architecture
set_target_properties(vector_add PROPERTIES
    CUDA_ARCHITECTURES "89"
)

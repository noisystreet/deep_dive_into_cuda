cmake_minimum_required(VERSION 3.18)
project(cuda_examples LANGUAGES CUDA CXX)

# Enable verbose output for nvcc (shows full command lines in build log)
# --keep preserves all intermediate compilation artifacts for analysis
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --verbose --keep")

# Optional: also show full compile/link command lines in Makefile output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set default CUDA architecture for all targets
set(CMAKE_CUDA_ARCHITECTURES "89")

# ===== 第 1 章：编译流程与中间产物 =====
add_executable(vector_add vector_add.cu)
add_executable(wmma_matmul wmma_matmul.cu)

# ===== 第 3 章：运行时 API =====
add_executable(graph_demo graph_demo.cu)
add_executable(streams_demo streams_demo.cu)
add_executable(memory_demo memory_demo.cu)
add_executable(uvm_pagefault_demo uvm_pagefault_demo.cu)

# ===== 第 4 章：进阶运行时专题 =====
add_executable(context_demo context_demo.cu)
target_link_libraries(context_demo PRIVATE CUDA::cuda_driver)

add_executable(greenctx_demo greenctx_demo.cu)
target_link_libraries(greenctx_demo PRIVATE CUDA::cuda_driver)

add_executable(graph_capture_demo graph_capture_demo.cu)
add_executable(module_demo module_demo.cu)
target_link_libraries(module_demo PRIVATE CUDA::cuda_driver)

add_executable(cublas_demo cublas_demo.cu)
target_link_libraries(cublas_demo PRIVATE CUDA::cublas)

add_executable(nvrtc_demo nvrtc_demo.cu)
target_link_libraries(nvrtc_demo PRIVATE CUDA::cuda_driver CUDA::nvrtc)

# ===== 多架构 / 分离编译示例 =====
add_executable(rdc_vector_add
    rdc/device_utils.cu
    rdc/vector_add_rdc.cu)
set_target_properties(rdc_vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

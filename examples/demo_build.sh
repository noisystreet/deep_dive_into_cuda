#!/bin/bash
# 演示编译过程中各中间产物的查看
# 使用 --keep 保留全部中间文件

NVCC_FLAGS="--verbose --keep -arch=sm_89"

echo "=== 1. 编译 vector_add (保留中间文件) ==="
nvcc $NVCC_FLAGS -o vector_add vector_add.cu

echo ""
echo "=== 2. 查看生成的中间文件 ==="
ls -lh *.ii *.ptx *.cubin *.fatbin *.cudafe1.* 2>/dev/null

echo ""
echo "=== 3. 查看 PTX (虚拟汇编) ==="
head -30 vector_add.ptx

echo ""
echo "=== 4. 运行程序 ==="
./vector_add

echo ""
echo "=========================================="
echo "=== 编译 wmma_matmul (保留中间文件) ==="
nvcc $NVCC_FLAGS -o wmma_matmul wmma_matmul.cu

echo ""
echo "=== 查看 WMMA 版本的 PTX ==="
grep 'wmma\.' wmma_matmul.ptx | head -10

echo ""
echo "=== 运行 WMMA 版本 ==="
./wmma_matmul

echo ""
echo "=========================================="
echo "=== 编译运行时专题示例 (第 3/4 章) ==="
# Driver API 示例需要 -lcuda
nvcc $NVCC_FLAGS -o context_demo context_demo.cu -lcuda
nvcc $NVCC_FLAGS -o greenctx_demo greenctx_demo.cu -lcuda
nvcc $NVCC_FLAGS -o module_demo module_demo.cu -lcuda
# cuBLAS 示例需要 -lcublas
nvcc $NVCC_FLAGS -o cublas_demo cublas_demo.cu -lcublas
# NVRTC 示例需要 -lnvrtc -lcuda
nvcc $NVCC_FLAGS -o nvrtc_demo nvrtc_demo.cu -lnvrtc -lcuda
# 仅依赖 Runtime 的示例
nvcc $NVCC_FLAGS -o memory_demo memory_demo.cu
nvcc $NVCC_FLAGS -o uvm_pagefault_demo uvm_pagefault_demo.cu
nvcc $NVCC_FLAGS -o graph_capture_demo graph_capture_demo.cu

echo ""
echo "=== 全部示例编译完成 ==="

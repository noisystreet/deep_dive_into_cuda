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

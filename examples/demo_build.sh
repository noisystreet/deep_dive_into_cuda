# 演示编译过程中各中间产物的查看
# 使用 --keep 保留全部中间文件

NVCC_FLAGS="--verbose --keep -arch=sm_89"
SRC="../examples/vector_add.cu"

echo "=== 1. 编译 CUDA 程序 (保留中间文件) ==="
nvcc $NVCC_FLAGS -o vector_add $SRC

echo ""
echo "=== 2. 查看生成的中间文件 ==="
ls -lh *.ii *.ptx *.cubin *.fatbin *.cudafe1.* 2>/dev/null

echo ""
echo "=== 3. 查看 PTX (虚拟汇编) ==="
head -30 vector_add.ptx

echo ""
echo "=== 4. 查看 cubin 结构 ==="
cuobjdump -elf vector_add.sm_89.cubin 2>/dev/null || echo "cuobjdump not available"

echo ""
echo "=== 5. 运行程序 ==="
./vector_add

echo ""
echo "=== 6. strace 跟踪驱动调用 ==="
strace -e ioctl,mmap,openat ./vector_add 2>&1 | grep -E 'nvidia|mmap' | head -10

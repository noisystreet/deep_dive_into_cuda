// memory_demo.cu - CUDA 内存管理各路径的系统调用追踪
// 编译: nvcc --verbose --keep -arch=sm_89 -o memory_demo memory_demo.cu
// 追踪: strace -f -e ioctl,mmap,openat,close,brk ./memory_demo 2>&1 | grep -E 'nvidia|mmap|ioctl'

#include <cuda_runtime.h>
#include <stdio.h>

const int MB = 1024 * 1024;
const int SIZE = 8 * MB;
const int LARGE_SIZE = 64 * MB;

void test_cudaMalloc() {
    printf("=== 1. cudaMalloc (GPU 显存) ===\n");
    float *d_ptr;
    cudaMalloc(&d_ptr, SIZE);
    cudaMemset(d_ptr, 0, SIZE);
    cudaFree(d_ptr);
}

void test_cudaHostAlloc() {
    printf("=== 2. cudaHostAlloc (Pinned Host) ===\n");
    float *h_ptr;
    cudaHostAlloc(&h_ptr, SIZE, cudaHostAllocDefault);
    for (int i = 0; i < SIZE / sizeof(float); i += 4096) h_ptr[i] = 1.0f;
    cudaFreeHost(h_ptr);
}

void test_cudaMallocManaged() {
    printf("=== 3. cudaMallocManaged (UVM) ===\n");
    float *m_ptr;
    cudaMallocManaged(&m_ptr, SIZE);
    m_ptr[0] = 1.0f;
    cudaDeviceSynchronize();
    cudaFree(m_ptr);
}

void test_cudaHostAlloc_mapped() {
    printf("=== 4. cudaHostAlloc Mapped (双映射) ===\n");
    float *h_ptr;
    cudaHostAlloc(&h_ptr, SIZE, cudaHostAllocMapped);
    float *d_ptr;
    cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);
    for (int i = 0; i < SIZE / sizeof(float); i += 4096) h_ptr[i] = 1.0f;
    cudaFreeHost(h_ptr);
}

void test_cudaMallocAsync() {
    printf("=== 5. cudaMallocAsync (流式显存池) ===\n");
    cudaStream_t s;
    cudaStreamCreate(&s);
    float *d_ptr;
    cudaMallocAsync(&d_ptr, SIZE, s);
    cudaMemsetAsync(d_ptr, 0, SIZE, s);
    cudaStreamSynchronize(s);
    cudaFreeAsync(d_ptr, s);
    cudaStreamDestroy(s);
}

void test_uvm_large() {
    printf("=== 6. UVM Large, 触发 page fault ===\n");
    float *m_ptr;
    cudaMallocManaged(&m_ptr, LARGE_SIZE);
    // CPU touch: triggers page fault → CPU allocation
    for (int i = 0; i < LARGE_SIZE / sizeof(float); i += 4096 / sizeof(float)) {
        m_ptr[i] = i;
    }
    // GPU kernel touch: triggers page fault → GPU migration
    cudaMemset(m_ptr, 0, LARGE_SIZE);
    cudaDeviceSynchronize();
    cudaFree(m_ptr);
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s | Total mem: %.0f MB\n", prop.name, prop.totalGlobalMem / (double)MB);

    test_cudaMalloc();
    test_cudaHostAlloc();
    test_cudaMallocManaged();
    test_cudaHostAlloc_mapped();
    test_cudaMallocAsync();
    test_uvm_large();
    printf("OK\n");
    return 0;
}

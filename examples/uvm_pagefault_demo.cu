// uvm_pagefault_demo.cu - UVM Page Fault 深入分析
// 编译: nvcc --verbose --keep -arch=sm_89 -o uvm_pagefault uvm_pagefault_demo.cu
// 追踪:
//   软缺页: strace -f -e ioctl,mmap ./uvm_pagefault 2>&1 | grep -E 'nvidia-uvm|0x4e|0x2a|0x2b'
//   性能:   nsys profile -t cuda,osrt -o uvm_report ./uvm_pagefault
//   内核:   sudo perf stat -e page-faults,dTLB-load-misses ./uvm_pagefault

#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

const int LARGE_SIZE = 512 * 1024 * 1024;   // 512 MB

// 辅助: 构造 cudaMemLocation (CUDA 13.1 API)
static cudaMemLocation loc_device(int dev) {
    cudaMemLocation l;
    l.type = cudaMemLocationTypeDevice;
    l.id = dev;
    return l;
}
static cudaMemLocation loc_host() {
    cudaMemLocation l;
    l.type = cudaMemLocationTypeHost;
    l.id = 0;
    return l;
}

void test_uvm_cpu_touch() {
    printf("=== 1. UVM CPU touch (惰性分配 + page fault) ===\n");
    float *m;
    cudaMallocManaged(&m, LARGE_SIZE);
    double t0 = now();
    for (int i = 0; i < LARGE_SIZE / sizeof(float); i += 4096 / sizeof(float)) {
        m[i] = i;
    }
    double t1 = now();
    printf("  CPU touch (stride=4096): %.3f s\n", t1 - t0);
    cudaFree(m);
}

void test_uvm_gpu_touch() {
    printf("=== 2. UVM GPU touch (page fault + 迁移) ===\n");
    float *m;
    cudaMallocManaged(&m, LARGE_SIZE);
    cudaMemset(m, 0, LARGE_SIZE);  // GPU 侧写入 → 页迁移到 GPU
    cudaDeviceSynchronize();
    double t0 = now();
    // CPU 读: 触发回迁 page fault
    volatile float tmp;
    for (int i = 0; i < LARGE_SIZE / sizeof(float); i += 4096 / sizeof(float)) {
        tmp = m[i];
    }
    (void)tmp;
    double t1 = now();
    printf("  GPU→CPU 回迁 (stride=4096): %.3f s\n", t1 - t0);
    cudaFree(m);
}

void test_uvm_prefetch() {
    printf("=== 3. UVM 预取 (无 page fault) ===\n");
    float *m;
    cudaMallocManaged(&m, LARGE_SIZE);
    cudaMemset(m, 0, LARGE_SIZE);
    cudaDeviceSynchronize();

    // 预取到 CPU
    double t0 = now();
    cudaMemPrefetchAsync(m, LARGE_SIZE, loc_host(), NULL);
    cudaDeviceSynchronize();
    double t1 = now();
    printf("  预取 GPU→CPU: %.3f s\n", t1 - t0);

    // CPU 读: 无 page fault
    t0 = now();
    volatile float tmp;
    for (int i = 0; i < LARGE_SIZE / sizeof(float); i += 4096 / sizeof(float)) {
        tmp = m[i];
    }
    (void)tmp;
    t1 = now();
    printf("  CPU 读 (预取后, 无 PF): %.3f s\n", t1 - t0);
    cudaFree(m);
}

void test_advise_preferred_location() {
    printf("=== 4. cudaMemAdvise: 设置首选位置 ===\n");
    float *m;
    cudaMallocManaged(&m, LARGE_SIZE);

    cudaMemAdvise(m, LARGE_SIZE, cudaMemAdviseSetPreferredLocation, loc_device(0));
    cudaMemAdvise(m, LARGE_SIZE, cudaMemAdviseSetAccessedBy, loc_host());

    // GPU touch — 在 GPU 端分配
    cudaMemset(m, 0, LARGE_SIZE);
    cudaDeviceSynchronize();

    double t0 = now();
    volatile float tmp;
    for (int i = 0; i < LARGE_SIZE / sizeof(float); i += 4096 / sizeof(float)) {
        tmp = m[i];
    }
    (void)tmp;
    double t1 = now();
    printf("  CPU 读 (advise 后): %.3f s\n", t1 - t0);
    cudaFree(m);
}

void test_uvm_stress() {
    printf("=== 5. UVM 压力: 批量小分配 + 触发 PF ===\n");
    const int N_ALLOCS = 8;
    float *ptrs[N_ALLOCS];
    for (int i = 0; i < N_ALLOCS; i++) {
        cudaMallocManaged(&ptrs[i], 64 * 1024 * 1024);  // 64 MB each
    }
    // 交错访问: 触发大量 page fault
    double t0 = now();
    for (int round = 0; round < 2; round++) {
        for (int i = 0; i < N_ALLOCS; i++) {
            for (int j = 0; j < 64 * 1024 * 1024 / sizeof(float); j += 4096 / sizeof(float)) {
                ptrs[i][j] = (float)(i + j);
            }
        }
    }
    double t1 = now();
    printf("  交错 8×64 MB: %.3f s\n", t1 - t0);
    for (int i = 0; i < N_ALLOCS; i++) cudaFree(ptrs[i]);
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s | UVM: %s\n", prop.name,
           prop.managedMemory ? "yes" : "no");

    test_uvm_cpu_touch();
    test_uvm_gpu_touch();
    test_uvm_prefetch();
    test_advise_preferred_location();
    test_uvm_stress();
    printf("Done\n");
    return 0;
}

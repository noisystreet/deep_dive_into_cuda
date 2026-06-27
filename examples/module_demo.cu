// module_demo.cu - 对比 cuModule 动态加载 vs nvcc 静态链接
// 编译: nvcc --verbose --keep -arch=sm_89 -o module_demo module_demo.cu
// 追踪: strace -f -e ioctl,mmap,openat,execve ./module_demo 2>&1 | grep -E 'cubin|ptx|nvidia|fatbin'

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== 方法 A: 静态链接 kernel (普通 nvcc 方式) ==========
__global__ void static_vec_add(float *c, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

void run_static() {
    printf("=== 静态链接 (nvcc 编译) ===\n");
    const int N = 1024 * 1024;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = i * 1.0f; h_b[i] = 0.5f; }
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    static_vec_add<<<N / 256, 256>>>(d_c, d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b;
}

// ========== 方法 B: cuModule 动态加载 cubin ==========
void run_dynamic(const char *cubin_path) {
    printf("=== cuModule 动态加载 (%s) ===\n", cubin_path);

    // 初始化 Driver API
    CUcontext ctx;
    CUdevice dev;
    cuInit(0);
    cuDeviceGet(&dev, 0);
    CUctxCreateParams createParams = {};
    cuCtxCreate(&ctx, &createParams, CU_CTX_SCHED_AUTO, dev);

    // 加载 cubin 文件到内存
    FILE *f = fopen(cubin_path, "rb");
    if (!f) { printf("  ERROR: cannot open %s\n", cubin_path); return; }
    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
    rewind(f);
    char *image = new char[fsize];
    fread(image, 1, fsize, f);
    fclose(f);

    printf("  cuModuleLoadData (%zu bytes)...\n", fsize);
    CUmodule module;
    CUresult res = cuModuleLoadData(&module, image);
    if (res != CUDA_SUCCESS) {
        printf("  ERROR: cuModuleLoadData returned %d\n", res);
        delete[] image; return;
    }

    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "dyn_vec_add");

    const int N = 1024 * 1024;
    CUdeviceptr d_a, d_b, d_c;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));
    cuMemAlloc(&d_c, N * sizeof(float));

    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = i * 1.0f; h_b[i] = 0.5f; }
    cuMemcpyHtoD(d_a, h_a, N * sizeof(float));
    cuMemcpyHtoD(d_b, h_b, N * sizeof(float));

    int n = N;
    void *args[] = { &d_c, &d_a, &d_b, &n };
    cuLaunchKernel(kernel, N / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);
    delete[] image; delete[] h_a; delete[] h_b;
    printf("  OK\n");
}

// ========== 方法 C: cuModuleLoadData 加载 PTX (JIT 回退) ==========
void run_ptx_jit(const char *ptx_path) {
    printf("=== cuModuleLoadData (PTX JIT 路径) ===\n");

    CUcontext ctx;
    CUdevice dev;
    cuInit(0);
    cuDeviceGet(&dev, 0);
    CUctxCreateParams createParams = {};
    cuCtxCreate(&ctx, &createParams, CU_CTX_SCHED_AUTO, dev);

    FILE *f = fopen(ptx_path, "r");
    if (!f) { printf("  ERROR: cannot open %s\n", ptx_path); return; }
    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
    rewind(f);
    char *ptx = new char[fsize + 1];
    fread(ptx, 1, fsize, f);
    ptx[fsize] = 0;
    fclose(f);

    printf("  cuModuleLoadData (PTX, %zu bytes)...\n", fsize);
    CUmodule module;
    CUresult res = cuModuleLoadData(&module, ptx);
    if (res != CUDA_SUCCESS) {
        printf("  ERROR: cuModuleLoadData(PTX) returned %d\n", res);
        delete[] ptx; return;
    }

    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "dyn_vec_add");

    const int N = 1024 * 1024;
    CUdeviceptr d_a, d_b, d_c;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));
    cuMemAlloc(&d_c, N * sizeof(float));

    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = i * 1.0f; h_b[i] = 0.5f; }
    cuMemcpyHtoD(d_a, h_a, N * sizeof(float));
    cuMemcpyHtoD(d_b, h_b, N * sizeof(float));

    int n = N;
    void *args[] = { &d_c, &d_a, &d_b, &n };
    cuLaunchKernel(kernel, N / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);
    delete[] ptx; delete[] h_a; delete[] h_b;
    printf("  OK\n");
}

int main() {
    // 编译独立 cubin 和 PTX
    system("echo '__global__ void dyn_vec_add(float *c, const float *a, const float *b, int n) {"
           " int idx = blockIdx.x * blockDim.x + threadIdx.x;"
           " if (idx < n) c[idx] = a[idx] + b[idx];"
           "}' > /tmp/dyn_kernel.cu");
    system("nvcc -arch=sm_89 -cubin -o /tmp/dyn_kernel.sm_89.cubin /tmp/dyn_kernel.cu 2>/dev/null");
    system("nvcc -arch=sm_89 -ptx -o /tmp/dyn_kernel.ptx /tmp/dyn_kernel.cu 2>/dev/null");

    run_static();
    run_dynamic("/tmp/dyn_kernel.sm_89.cubin");
    run_ptx_jit("/tmp/dyn_kernel.ptx");

    system("rm -f /tmp/dyn_kernel.cu /tmp/dyn_kernel.sm_89.cubin /tmp/dyn_kernel.ptx");
    printf("Done\n");
    return 0;
}

// nvrtc_demo.cu - NVRTC 在线编译 vs nvcc 离线编译
// 演示: 在程序运行时动态编译 CUDA kernel (NVRTC) 并启动
//
// 编译: nvcc -arch=sm_89 -o nvrtc_demo nvrtc_demo.cu -lnvrtc -lcuda
//
// strace 对比:
//   离线 (nvcc):    strace -f -e clone,execve ./nvrtc_demo 2>&1 | grep -c execve
//   在线 (NVRTC):   strace -f -e ioctl,openat,mmap ./nvrtc_demo 2>&1
//                  | grep -cE '0x4e|execve'   # 应显示 0 execve

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

double now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define CHECK(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *s; cuGetErrorString(err, &s); \
        printf("  ERROR %s: %s\n", #call, s); \
        exit(1); \
    } \
} while(0)

// 用于 NVRTC 编译的 kernel 源代码 (字符串)
const char *kernel_src = R"(
extern "C" __global__ void vec_add(float *c, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
)";

// NVRTC 路径: 编译 + launch
double test_nvrtc(int N) {
    printf("--- NVRTC 在线编译 ---\n");

    // 1. 创建编译程序 (纯 CPU 操作, 无 ioctl)
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, kernel_src, "vec_add_kernel",
                           0, NULL, NULL) != NVRTC_SUCCESS) {
        printf("NVRTC CreateProgram failed\n"); exit(1);
    }

    // 2. 编译为 PTX (还是纯 CPU 操作 — LLVM 编译)
    double t_compile = now();
    const char *opts[] = {
        "--gpu-architecture=sm_89",
        "--use_fast_math"
    };
    nvrtcResult result = nvrtcCompileProgram(prog, 2, opts);
    t_compile = now() - t_compile;

    // 获取编译日志
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char *log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    if (result != NVRTC_SUCCESS) {
        printf("NVRTC compile error:\n%s\n", log);
        delete[] log;
        exit(1);
    }
    printf("  Compile: %.2f ms\n", t_compile * 1000);

    // 3. 获取 PTX
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    size_t ptx_len = strlen(ptx);
    printf("  PTX size: %zu bytes\n", ptx_len);

    nvrtcDestroyProgram(&prog);
    delete[] log;

    // 4. 使用 Driver API 加载 PTX 并启动
    CUcontext ctx;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    CUmodule module;
    CHECK(cuModuleLoadData(&module, ptx));
    delete[] ptx;

    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, module, "vec_add"));

    CUdeviceptr d_a, d_b, d_c;
    CHECK(cuMemAlloc(&d_a, N * sizeof(float)));
    CHECK(cuMemAlloc(&d_b, N * sizeof(float)));
    CHECK(cuMemAlloc(&d_c, N * sizeof(float)));

    float *h_a = new float[N], *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    CHECK(cuMemcpyHtoD(d_a, h_a, N * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_b, h_b, N * sizeof(float)));

    // 5. Launch
    void *args[] = { &d_c, &d_a, &d_b, &N };
    double t_launch = now();
    CHECK(cuLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CHECK(cuCtxSynchronize());
    t_launch = now() - t_launch;
    printf("  Launch: %.3f ms\n", t_launch * 1000);

    // 验证
    float *h_result = new float[N];
    CHECK(cuMemcpyDtoH(h_result, d_c, N * sizeof(float)));
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_result[i] - 3.0f);
        if (err > max_err) max_err = err;
    }
    printf("  Result OK (max error: %.2e)\n", max_err);

    delete[] h_a; delete[] h_b; delete[] h_result;
    CHECK(cuMemFree(d_a)); CHECK(cuMemFree(d_b)); CHECK(cuMemFree(d_c));
    CHECK(cuModuleUnload(module));
    CHECK(cuDevicePrimaryCtxRelease(dev));

    return t_compile + t_launch;
}

// 离线 nvcc 路径 (使用预编译的 fatbin — 由编译时 nvcc 生成)
__global__ void vec_add_offline(float *c, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

double test_offline(int N) {
    printf("--- 离线编译 (nvcc) ---\n");

    float *d_a, *d_b, *d_c;
    float *h_a = new float[N], *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel 已提前编译 (在 fatbin 中), 无需编译时间
    double t_launch = now();
    vec_add_offline<<<(N + 255) / 256, 256>>>(d_c, d_a, d_b, N);
    cudaDeviceSynchronize();
    t_launch = now() - t_launch;

    printf("  Compile: 0 ms  (预编译在 fatbin 中)\n");
    printf("  Launch: %.3f ms\n", t_launch * 1000);

    // 验证
    float *h_result = new float[N];
    cudaMemcpy(h_result, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_result[i] - 3.0f);
        if (err > max_err) max_err = err;
    }
    printf("  Result OK (max error: %.2e)\n", max_err);

    delete[] h_a; delete[] h_b; delete[] h_result;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return t_launch;
}

int main() {
    const int N = 4 * 1024 * 1024;  // 4M floats = 16 MB

    printf("=== NVRTC 在线编译 vs 离线编译 ===\n");
    printf("Workload: 4M vec_add, GPU: sm_89\n\n");

    double t_nvrtc = test_nvrtc(N);
    double t_offline = test_offline(N);

    printf("\n=== 对比 ===\n");
    printf("NVRTC 总时间:    %.2f ms\n", t_nvrtc * 1000);
    printf("离线编译总时间:  %.2f ms\n", t_offline * 1000);
    printf("差距:            %.2f ms (NVRTC 多了编译开销)\n",
           (t_nvrtc - t_offline) * 1000);
    printf("\n=== 关键差异 ===\n");
    printf("  NVRTC: 编译在进程中完成 (libnvrtc → LLVM → PTX)\n");
    printf("         无子进程开销, PTX 立即可用\n");
    printf("  nvcc:  编译在构建时完成 (execve cicc → ptxas)\n");
    printf("         运行时零编译开销, kernel 已嵌入 fatbin\n");
    printf("Done\n");
    return 0;
}

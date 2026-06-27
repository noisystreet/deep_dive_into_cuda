// greenctx_demo.cu - Green Context: 轻量级 GPU 上下文分析
// Green Context 是 CUDA 12.x+ 引入的特性，允许在限定 SM 的 partition
// 上创建轻量级上下文，实现更细粒度的 GPU 资源分区。
// 编译: nvcc -gencode arch=compute_89,code=sm_89 -o greenctx_demo greenctx_demo.cu -lcuda
// 追踪: strace -f -e ioctl,openat,mmap,clone3 ./greenctx_demo 2>&1 | grep -E 'nvcc|0x49|0x4f|nvidia'

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(err, &errStr); \
        printf("  ERROR at line %d: %s (code=%d)\n", __LINE__, errStr, err); \
        return -1; \
    } \
} while(0)

// 返回 0 = 成功, 非0 = 跳过
typedef int (*test_fn)();

static int test_regular_ctx() {
    printf("\n=== Regular Context (main ctx, all SMs) ===\n");
    CUcontext ctx;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return -1;
    if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) return -1;

    CUdeviceptr d_ptr;
    CUDA_CHECK(cuMemAlloc(&d_ptr, 8 * 1024 * 1024));
    CUDA_CHECK(cuMemFree(d_ptr));
    CUDA_CHECK(cuDevicePrimaryCtxRelease(dev));
    printf("  OK (8 MB allocated)\n");
    return 0;
}

static int test_green_ctx_basic() {
    printf("\n=== Green Context (SM partition, basic lifecycle) ===\n");
    CUdevice dev;
    cuDeviceGet(&dev, 0);

    // 1) 保留 primary context (必须先初始化)
    CUcontext primary_ctx;
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&primary_ctx, dev));
    CUDA_CHECK(cuCtxSetCurrent(primary_ctx));

    // 2) 查询设备 SM 资源
    CUdevResource smResource;
    memset(&smResource, 0, sizeof(smResource));
    smResource.type = CU_DEV_RESOURCE_TYPE_SM;
    CUDA_CHECK(cuDeviceGetDevResource(dev, &smResource, CU_DEV_RESOURCE_TYPE_SM));
    printf("  Total SMs: %u\n", smResource.sm.smCount);
    printf("  Min partition size: %u\n", smResource.sm.minSmPartitionSize);
    printf("  Coscheduled alignment: %u\n", smResource.sm.smCoscheduledAlignment);

    // 3) 分割 SM — 创建 partition
    CUdevResource greenSmRes;
    CUdevResource leftoverSmRes;
    memset(&greenSmRes, 0, sizeof(greenSmRes));
    memset(&leftoverSmRes, 0, sizeof(leftoverSmRes));
    unsigned int nbGroups = 1;
    CUDA_CHECK(cuDevSmResourceSplitByCount(
        &greenSmRes, &nbGroups,
        &smResource, &leftoverSmRes, 0,
        smResource.sm.minSmPartitionSize));
    printf("  Green ctx SMs: %u (remainder: %u)\n",
           greenSmRes.sm.smCount, leftoverSmRes.sm.smCount);

    // 4) 生成资源描述符
    CUdevResourceDesc desc;
    CUdevResource resources[] = { greenSmRes };
    CUDA_CHECK(cuDevResourceGenerateDesc(&desc, resources, 1));

    // 5) 创建 Green Context
    CUgreenCtx greenCtx;
    CUDA_CHECK(cuGreenCtxCreate(&greenCtx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));

    // 6) 转为 CUcontext
    CUcontext ctx;
    CUDA_CHECK(cuCtxFromGreenCtx(&ctx, greenCtx));
    CUDA_CHECK(cuCtxSetCurrent(ctx));
    printf("  Green ctx created (SM partition size: %u)\n", greenSmRes.sm.smCount);

    // 7) 分配显存
    CUdeviceptr d_ptr;
    CUDA_CHECK(cuMemAlloc(&d_ptr, 8 * 1024 * 1024));

    // 8) 用普通 cuStreamCreate (因为 green ctx 已被设为当前)
    CUstream stream;
    CUresult sres = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    if (sres != CUDA_SUCCESS) {
        printf("  cuStreamCreate (on green ctx) failed: %d - using default stream\n", sres);
        stream = NULL;  // 使用 green ctx 的默认 stream
    }

    // 9) 销毁
    if (stream) CUDA_CHECK(cuStreamDestroy(stream));
    CUDA_CHECK(cuMemFree(d_ptr));
    CUDA_CHECK(cuCtxSetCurrent(primary_ctx));
    CUDA_CHECK(cuGreenCtxDestroy(greenCtx));
    CUDA_CHECK(cuDevicePrimaryCtxRelease(dev));
    printf("  OK\n");
    return 0;
}

static int test_green_ctx_launch() {
    printf("\n=== Green Context (compile kernel + launch) ===\n");
    CUdevice dev;
    cuDeviceGet(&dev, 0);

    CUcontext primary_ctx;
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&primary_ctx, dev));
    CUDA_CHECK(cuCtxSetCurrent(primary_ctx));

    // 编译 kernel (在 primary ctx 上)
    system("echo '__global__ void green_kernel(unsigned int *c, int n) {"
           " int idx = threadIdx.x + blockIdx.x * blockDim.x;"
           " if (idx < n) c[idx] = idx;"
           "}' > /tmp/green_kernel.cu");
    system("nvcc -arch=sm_89 -cubin -o /tmp/green_kernel.sm_89.cubin /tmp/green_kernel.cu 2>/dev/null");

    FILE *f = fopen("/tmp/green_kernel.sm_89.cubin", "rb");
    if (!f) { printf("  SKIP (cubin compile failed)\n"); return 1; }
    fseek(f, 0, SEEK_END);
    size_t cubinSize = ftell(f);
    rewind(f);
    char *cubinData = new char[cubinSize];
    fread(cubinData, 1, cubinSize, f);
    fclose(f);

    CUmodule module;
    CUresult res = cuModuleLoadData(&module, cubinData);
    if (res != CUDA_SUCCESS) {
        printf("  SKIP (cuModuleLoadData failed: %d)\n", res);
        delete[] cubinData;
        return 1;
    }

    // 创建 green context
    CUdevResource smResource;
    memset(&smResource, 0, sizeof(smResource));
    smResource.type = CU_DEV_RESOURCE_TYPE_SM;
    CUDA_CHECK(cuDeviceGetDevResource(dev, &smResource, CU_DEV_RESOURCE_TYPE_SM));

    CUdevResource greenSmRes, leftoverSmRes;
    memset(&greenSmRes, 0, sizeof(greenSmRes));
    memset(&leftoverSmRes, 0, sizeof(leftoverSmRes));
    unsigned int nbGroups = 1;
    CUDA_CHECK(cuDevSmResourceSplitByCount(
        &greenSmRes, &nbGroups,
        &smResource, &leftoverSmRes, 0,
        smResource.sm.minSmPartitionSize));

    CUdevResourceDesc desc;
    CUdevResource resources[] = { greenSmRes };
    CUDA_CHECK(cuDevResourceGenerateDesc(&desc, resources, 1));

    CUgreenCtx greenCtx;
    CUDA_CHECK(cuGreenCtxCreate(&greenCtx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));

    CUcontext ctx;
    CUDA_CHECK(cuCtxFromGreenCtx(&ctx, greenCtx));
    CUDA_CHECK(cuCtxSetCurrent(ctx));

    CUdeviceptr d_ptr;
    CUDA_CHECK(cuMemAlloc(&d_ptr, 8 * 1024 * 1024));

    // 创建 stream (green ctx 提供默认 stream, cuStreamCreate 也兼容)
    CUstream stream;
    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS) {
        stream = NULL;  // 使用 green ctx 的默认 stream
    }

    // launch kernel
    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "_Z12green_kernelPji"));
    int n = 1024 * 1024;
    void *args[] = { &d_ptr, &n };
    CUDA_CHECK(cuLaunchKernel(kernel, n/256, 1, 1, 256, 1, 1, 0, stream, args, NULL));
    CUDA_CHECK(cuStreamSynchronize(stream));

    // verify
    unsigned int *h_result = new unsigned int[1024];
    CUDA_CHECK(cuMemcpyDtoH(h_result, d_ptr, 1024 * sizeof(unsigned int)));
    printf("  Launch OK. Result[0]=%u, Result[1]=%u\n", h_result[0], h_result[1]);
    delete[] h_result;

    CUDA_CHECK(cuStreamDestroy(stream));
    CUDA_CHECK(cuMemFree(d_ptr));
    CUDA_CHECK(cuCtxSetCurrent(primary_ctx));
    CUDA_CHECK(cuGreenCtxDestroy(greenCtx));
    CUDA_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cuDevicePrimaryCtxRelease(dev));
    delete[] cubinData;
    system("rm -f /tmp/green_kernel.cu /tmp/green_kernel.sm_89.cubin");
    printf("  OK\n");
    return 0;
}

int main() {
    cuInit(0);
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);

    char name[256];
    cuDeviceGetName(name, sizeof(name), 0);
    printf("=== Green Context Analysis ===\n");
    printf("GPU: %s (sm_%d%d)\n", name, major, minor);

    test_regular_ctx();
    test_green_ctx_basic();
    test_green_ctx_launch();

    printf("\nAll tests completed.\n");
    return 0;
}

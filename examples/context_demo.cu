// context_demo.cu - CUDA Context 创建/切换/销毁的系统调用追踪
// 编译: nvcc -arch=sm_89 -o context_demo context_demo.cu -lcuda
// 追踪: strace -f -e ioctl,mmap,openat,close ./context_demo 2>&1 | grep -E 'nvidia|NV_DEV|0x2b|0x2a|0x49'

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

void test_cuInit() {
    printf("=== 1. cuInit (Driver 初始化) ===\n");
    CUresult res = cuInit(0);
    printf("  cuInit(0) = %d\n", res);
}

void test_cuCtxCreate() {
    printf("=== 2. cuCtxCreate (创建上下文) ===\n");
    CUcontext ctx;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUctxCreateParams params = {};
    CUresult res = cuCtxCreate(&ctx, &params, CU_CTX_SCHED_AUTO, dev);
    printf("  cuCtxCreate = %d, ctx=%p\n", res, (void*)ctx);

    // 在当前上下文中分配显存
    CUdeviceptr d_ptr;
    cuMemAlloc(&d_ptr, 8 * 1024 * 1024);
    printf("  cuMemAlloc(8 MB) = %p\n", (void*)d_ptr);
    cuMemFree(d_ptr);
    cuCtxDestroy(ctx);
}

void test_cuCtxPushPop() {
    printf("=== 3. cuCtxPush/Pop (上下文切换) ===\n");
    CUcontext ctx1, ctx2;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUctxCreateParams params = {};
    cuCtxCreate(&ctx1, &params, CU_CTX_SCHED_AUTO, dev);
    cuCtxCreate(&ctx2, &params, CU_CTX_SCHED_AUTO, dev);

    // 切换上下文栈
    cuCtxPushCurrent(ctx1);
    CUcontext cur;
    cuCtxGetCurrent(&cur);
    printf("  Push ctx1, current=%p vs ctx1=%p\n", (void*)cur, (void*)ctx1);

    // 在 ctx1 中分配
    CUdeviceptr d1;
    cuMemAlloc(&d1, 4 * 1024 * 1024);

    cuCtxPushCurrent(ctx2);
    cuCtxGetCurrent(&cur);
    printf("  Push ctx2, current=%p vs ctx2=%p\n", (void*)cur, (void*)ctx2);

    // 在 ctx2 中分配
    CUdeviceptr d2;
    cuMemAlloc(&d2, 4 * 1024 * 1024);

    // 切换回 ctx1 (pop ctx2)
    ;
    printf("  Pop => current=%p vs ctx1=%p\n", (void*)cur, (void*)ctx1);

    cuMemFree(d1);
    cuMemFree(d2);
    cuCtxDestroy(ctx1);
    cuCtxDestroy(ctx2);
}

void test_multi_context_concurrent() {
    printf("=== 4. 多上下文并发 launch ===\n");
    const int N = 1024 * 1024;
    const int N_CTX = 4;
    CUcontext ctxs[N_CTX];
    CUdeviceptr ptrs[N_CTX];

    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUctxCreateParams params = {};

    for (int i = 0; i < N_CTX; i++) {
        cuCtxCreate(&ctxs[i], &params, CU_CTX_SCHED_AUTO, dev);
        cuCtxPushCurrent(ctxs[i]);
        cuMemAlloc(&ptrs[i], N * sizeof(float));
        ;
        // need to get current for pop
    }

    printf("  创建 %d 个上下文，每个 4 MB 显存\n", N_CTX);

    // 每个上下文 launch kernel
    // 这里简化为在不同上下文间切换并 launch
    for (int i = 0; i < N_CTX; i++) {
        cuCtxSetCurrent(ctxs[i]);
        cuCtxSynchronize();
    }

    for (int i = 0; i < N_CTX; i++) {
        cuCtxSetCurrent(ctxs[i]);
        cuMemFree(ptrs[i]);
        cuCtxDestroy(ctxs[i]);
    }
}

void test_cuMemHostAlloc_pinned() {
    printf("=== 5. cuMemAllocHost (Pinned Memory) ===\n");
    CUcontext ctx;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUctxCreateParams params = {};
    cuCtxCreate(&ctx, &params, CU_CTX_SCHED_AUTO, dev);

    void *h_ptr;
    cuMemAllocHost(&h_ptr, 8 * 1024 * 1024);
    // Touch pages
    for (int i = 0; i < 8 * 1024 * 1024; i += 4096) {
        ((volatile char*)h_ptr)[i] = 1;
    }
    cuMemFreeHost(h_ptr);
    cuCtxDestroy(ctx);
}

int main() {
    printf("=== CUDA Context 深度分析 ===\n\n");

    test_cuInit();
    test_cuCtxCreate();
    test_cuCtxPushPop();
    test_multi_context_concurrent();
    test_cuMemHostAlloc_pinned();

    printf("\nDone\n");
    return 0;
}

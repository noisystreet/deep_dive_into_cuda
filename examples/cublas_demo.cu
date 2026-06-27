// cublas_demo.cu - cuBLAS 库调用链分析
// 对比: 手写 naive GEMM vs cuBLAS SGEMM 的 launch 行为差异
//
// 编译: nvcc -arch=sm_89 -o cublas_demo cublas_demo.cu -lcublas
// 追踪: strace -f -e ioctl,openat,mmap,writev ./cublas_demo 2>&1
//       | grep -E '0x4e|0x2a|0x2b|nvidia|cuBLAS|cublas' | head -40
// ld统计: LD_DEBUG=libs ./cublas_demo 2>&1 | grep cublas

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ===== 手写 naive GEMM (验证用, 性能很差) =====
__global__ void naive_sgemm(float *C, const float *A, const float *B,
                            int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ===== cuBLAS SGEMM =====
void run_cublas(cublasHandle_t handle, float *d_C, float *d_A, float *d_B,
                int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
}

int main() {
    int M = 1024, N = 1024, K = 1024;  // 1K×1K GEMM
    printf("=== cuBLAS SGEMM 调用链分析 ===\n");
    printf("Matrix: %dx%d (A) × %dx%d (B) = %dx%d (C)\n", M, K, K, N, M, N);
    printf("Size: A=%.1f MB, B=%.1f MB, C=%.1f MB\n\n",
           M * K * 4 / 1e6, K * N * 4 / 1e6, M * N * 4 / 1e6);

    // 分配 host 内存
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    printf("  cudaMalloc done: %.1f MB total\n",
           3 * (M * K * 4) / 1e6);

    // ===== 1. 手写 naive GEMM =====
    printf("\n--- 1. 手写 naive GEMM ---\n");
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    naive_sgemm<<<grid, block>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    cudaDeviceSynchronize();

    double t0 = now();
    for (int i = 0; i < 10; i++) {
        naive_sgemm<<<grid, block>>>(d_C, d_A, d_B, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();
    double t1 = now();
    printf("  10 launches: %.3f ms (avg %.2f ms)\n",
           (t1 - t0) * 1000, (t1 - t0) * 100);

    // ===== 2. cuBLAS SGEMM =====
    printf("\n--- 2. cuBLAS SGEMM ---\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warmup
    run_cublas(handle, d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    t0 = now();
    for (int i = 0; i < 10; i++) {
        run_cublas(handle, d_C, d_A, d_B, M, N, K);
    }
    cudaDeviceSynchronize();
    t1 = now();
    printf("  10 launches: %.3f ms (avg %.2f ms)\n",
           (t1 - t0) * 1000, (t1 - t0) * 100);

    // ===== 3. cuBLAS 结果验证 =====
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n--- 3. cuBLAS 结果校验 ---\n");
    float max_err = 0;
    for (int i = 0; i < min(100, M * N); i++) {
        float expected = 0;
        int row = i / N, col = i % N;
        for (int k = 0; k < K; k++)
            expected += h_A[row * K + k] * h_B[k * N + col];
        float err = fabs(h_C[i] - expected) / fabs(expected);
        if (err > max_err) max_err = err;
    }
    printf("  Max relative error (first 100 elements): %.2e\n", max_err);

    // ===== 4. cuBLAS library 信息 =====
    printf("\n--- 4. cuBLAS 版本信息 ---\n");
    int ver;
    cublasGetVersion(handle, &ver);
    printf("  cuBLAS version: %d\n", ver);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    printf("\nDone.\n");
    return 0;
}

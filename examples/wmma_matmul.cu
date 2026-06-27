// wmma_matmul.cu - WMMA (Warp Matrix Multiply-Accumulate) 矩阵乘法
// 编译: nvcc --verbose --keep -arch=sm_89 -o wmma_matmul wmma_matmul.cu
// 观察: PTX/SASS 中 HMMA 指令的生成

#include <cuda_runtime.h>
#include <crt/mma.h>
#include <stdio.h>

// WMMA 矩阵乘法: C = A * B
// A: m×k (row_major half), B: k×n (col_major half), C: m×n (row_major float)
// 每个 warp 处理一个 16×16 输出 tile
__global__ void wmma_matmul(half *A, half *B, float *C,
                            int M, int N, int K) {
    // WMMA 片段: 16×16×16
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                           nvcuda::wmma::row_major> frag_A;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                           nvcuda::wmma::col_major> frag_B;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_C;

    nvcuda::wmma::fill_fragment(frag_C, 0.0f);

    // warp 在输出矩阵中的位置
    int warp_m = blockIdx.x;  // 每个 block 一个 warp
    int warp_n = blockIdx.y;

    // 沿 K 维度以 16 为步长累加
    for (int k = 0; k < K; k += 16) {
        nvcuda::wmma::load_matrix_sync(frag_A, A + warp_m * 16 * K + k, K);
        nvcuda::wmma::load_matrix_sync(frag_B, B + k * N + warp_n * 16, N);
        // Tensor Core 矩阵乘: D = A * B + C
        nvcuda::wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
    }

    // 写回 16×16 结果
    nvcuda::wmma::store_matrix_sync(C + warp_m * 16 * N + warp_n * 16,
                                    frag_C, N, nvcuda::wmma::mem_row_major);
}

// 普通全局内存矩阵乘法 (对照基准)
__global__ void naive_matmul(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 16, N = 16, K = 16;

    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    float *h_C = new float[M * N]();

    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)(i % 4));

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    wmma_matmul<<<dim3(1,1), 32>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    float max_error = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += __half2float(h_A[i * K + k]) *
                            __half2float(h_B[k * N + j]);
            }
            float error = h_C[i * N + j] - expected;
            if (error < 0) error = -error;
            if (error > max_error) max_error = error;
        }
    }
    printf("WMMA MatMul %dx%d K=%d\n", M, N, K);
    printf("  Max error: %f\n", max_error);
    printf("  C[0][0] = %f\n", h_C[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

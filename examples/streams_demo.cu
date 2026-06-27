// streams_demo.cu - 追踪 CUDA Stream 的 Runtime → Driver → 内核路径
// 编译: nvcc --verbose --keep -arch=sm_89 -o streams_demo streams_demo.cu
// 追踪: strace -e ioctl,openat,mmap,clone3,eventfd2 ./streams_demo 2>&1 | grep nvidia | head -60

#include <cuda_runtime.h>
#include <stdio.h>

const int N = 64 * 1024 * 1024;  // 64M elements (256 MB)
const int BLOCK = 256;

__global__ void vec_add(float *c, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void vec_mul(float *c, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

int main() {
    float *d_a, *d_b, *d_c1, *d_c2;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c1, N * sizeof(float));
    cudaMalloc(&d_c2, N * sizeof(float));

    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = i * 1.0f; h_b[i] = (i % 100) * 1.0f; }
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    printf("=== 默认 stream: vec_add (同步) ===\n");
    vec_add<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_c1, d_a, d_b, N);
    cudaDeviceSynchronize();

    printf("=== 两个 stream 并发: vec_add | vec_mul ===\n");
    vec_add<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s1>>>(d_c1, d_a, d_b, N);
    vec_mul<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s2>>>(d_c2, d_a, d_b, N);
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);

    printf("=== stream 间依赖: s2 等待 s1 完成 ===\n");
    vec_add<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s1>>>(d_c1, d_a, d_b, N);
    cudaStreamWaitEvent(s2, NULL);  // s2 waits for s1 (null event = all prior work)
    vec_mul<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s2>>>(d_c2, d_a, d_b, N);
    cudaDeviceSynchronize();

    printf("=== Event 标记 ===\n");
    cudaEvent_t e1;
    cudaEventCreate(&e1);
    vec_add<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s1>>>(d_c1, d_a, d_b, N);
    cudaEventRecord(e1, s1);
    vec_mul<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s2>>>(d_c2, d_a, d_b, N);
    cudaStreamWaitEvent(s2, e1);  // s2 waits for e1 from s1
    cudaDeviceSynchronize();

    printf("=== 默认 stream 阻塞语义 ===\n");
    // 默认流是同步的：它等待所有非阻塞流完成，反之亦然
    vec_add<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, s1>>>(d_c1, d_a, d_b, N);
    vec_mul<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>        (d_c2, d_a, d_b, N);  // 默认流 - 阻塞
    // 这里默认流必须等 s1 完成，s1 也必须等默认流完成（隐式同步）
    cudaDeviceSynchronize();

    cudaEventDestroy(e1);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c1); cudaFree(d_c2);
    delete[] h_a; delete[] h_b;
    printf("OK\n");
    return 0;
}

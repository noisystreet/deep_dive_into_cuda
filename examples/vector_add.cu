#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel: vector addition
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;  // 1M elements
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    // Initialize input data
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        float error = h_c[i] - expected;
        if (error < 0) error = -error;
        if (error > max_error) max_error = error;
    }
    printf("Max error: %f\n", max_error);
    printf("Result: h_c[0] = %f (expected %f)\n", h_c[0], h_a[0] + h_b[0]);
    printf("Result: h_c[%d] = %f (expected %f)\n", n - 1, h_c[n - 1], h_a[n - 1] + h_b[n - 1]);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
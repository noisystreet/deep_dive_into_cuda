#include <cstdio>
#include <cuda_runtime.h>

__device__ float rdc_add(float a, float b);

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = rdc_add(a[idx], b[idx]);
    }
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        const float expected = h_a[i] + h_b[i];
        float error = h_c[i] - expected;
        if (error < 0) error = -error;
        if (error > max_error) max_error = error;
    }

    printf("RDC vector_add max error: %f\n", max_error);
    printf("h_c[0] = %f (expected %f)\n", h_c[0], h_a[0] + h_b[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return max_error > 1e-5f;
}

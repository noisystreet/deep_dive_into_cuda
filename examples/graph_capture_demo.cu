// graph_capture_demo.cu - CUDA Graph 捕获/实例化/launch 系统调用追踪
// 对比三种执行模式: 普通 launch / Graph instantiate+launch / Graph 重放
//
// 编译: nvcc -arch=sm_89 -o graph_capture graph_capture_demo.cu
// 追踪: strace -f -e ioctl,openat,mmap,clone3 ./graph_capture 2>&1
//       | grep -E '0x4e|0x4f|0x2a|0x2b|fatbin|nvcc|cic|ptx|execve'
// nsys:  nsys profile -t cuda,osrt -o graph_report ./graph_capture

#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

const int N = 32 * 1024 * 1024;  // 32M float — 保证 kernel 执行时间 > launch 开销
const int BLOCK = 256;

// 简单的 vector add kernel
__global__ void vec_add(float *c, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// 模式 A: 普通 launch
double test_normal_launch(cudaStream_t stream, float *d_c, float *d_a, float *d_b, int iterations) {
    double t0 = now();
    for (int i = 0; i < iterations; i++) {
        vec_add<<<N / BLOCK, BLOCK, 0, stream>>>(d_c, d_a, d_b, N);
    }
    cudaStreamSynchronize(stream);
    double t1 = now();
    return t1 - t0;
}

// 模式 B: Graph capture → instantiate → launch
double test_graph_capture(cudaStream_t stream, float *d_c, float *d_a, float *d_b, int iterations) {
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // 开始捕获
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    vec_add<<<N / BLOCK, BLOCK, 0, stream>>>(d_c, d_a, d_b, N);
    cudaStreamEndCapture(stream, &graph);

    // 实例化
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // 首次 launch (warmup)
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);

    // 计时: 连续重放
    double t0 = now();
    for (int i = 0; i < iterations; i++) {
        cudaGraphLaunch(instance, stream);
    }
    cudaStreamSynchronize(stream);
    double t1 = now();

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    return t1 - t0;
}

// 模式 C: 多次 Graph capture + instantiate (模拟动态图场景)
double test_multi_graph(cudaStream_t stream, float *d_c, float *d_a, float *d_b, float *d_a2, int iterations) {
    double t0 = now();
    for (int i = 0; i < iterations; i++) {
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        // 每次使用不同的输入指针 (模拟动态图)
        float *input = (i % 2 == 0) ? d_a : d_a2;
        vec_add<<<N / BLOCK, BLOCK, 0, stream>>>(d_c, input, d_b, N);
        cudaStreamEndCapture(stream, &graph);

        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);

        cudaGraphExecDestroy(instance);
        cudaGraphDestroy(graph);
    }
    double t1 = now();
    return t1 - t0;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *d_a, *d_a2, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_a2, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a2, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_a; delete[] h_b;

    const int ITER = 100;

    printf("=== CUDA Graph 系统调用对比 ===\n");
    printf("GPU memory: %d MB / kernel: vec_add / iterations: %d\n\n",
           (int)(4 * N * sizeof(float) / 1024 / 1024), ITER);

    // Warmup
    vec_add<<<N / BLOCK, BLOCK, 0, stream>>>(d_c, d_a, d_b, N);
    cudaStreamSynchronize(stream);

    double t_normal = test_normal_launch(stream, d_c, d_a, d_b, ITER);
    printf("模式 A - 普通 launch (×%d):  %.3f ms (avg %.3f us/launch)\n",
           ITER, t_normal * 1000, t_normal / ITER * 1e6);

    double t_graph = test_graph_capture(stream, d_c, d_a, d_b, ITER);
    printf("模式 B - Graph 重放 (×%d):   %.3f ms (avg %.3f us/launch)\n",
           ITER, t_graph * 1000, t_graph / ITER * 1e6);

    double t_multi = test_multi_graph(stream, d_c, d_a, d_b, d_a2, 10);
    printf("模式 C - 多次 Graph 重建 (×10): %.3f ms (avg %.3f ms/capture)\n",
           t_multi * 1000, t_multi / 10 * 1000);

    printf("\n加速比 (普通 / Graph):  %.2f×\n", t_normal / t_graph);

    cudaFree(d_a); cudaFree(d_a2); cudaFree(d_b); cudaFree(d_c);
    cudaStreamDestroy(stream);
    printf("Done\n");
    return 0;
}

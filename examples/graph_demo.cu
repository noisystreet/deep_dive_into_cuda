// graph_demo.cu - 追踪 CUDA Graph 的创建/实例化/启动路径
// 编译: nvcc --verbose --keep -arch=sm_89 -o graph_demo graph_demo.cu
// 追踪: strace -f -e ioctl,openat,clone3,execve ./graph_demo 2>&1 | grep -E 'nvidia|ptxas|cicc|nvlink' | head -80

#include <cuda_runtime.h>
#include <stdio.h>

const int N = 1024 * 1024;
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
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = i * 1.0f; h_b[i] = (i % 100) * 1.0f; }
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 方法 1: 普通 kernel launch（基准）
    printf("=== 普通 launch ===\n");
    vec_add<<<N / BLOCK, BLOCK>>>(d_c, d_a, d_b, N);
    cudaDeviceSynchronize();

    // 方法 2: CUDA Graph - 捕获模式
    printf("=== Graph: 捕获启动 ===\n");
    cudaGraph_t graph;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 开始捕获: 后续此 stream 上的 API 调用被记录到 graph 而非立即执行
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    vec_add<<<N / BLOCK, BLOCK, 0, stream>>>(d_c, d_a, d_b, N);
    vec_mul<<<N / BLOCK, BLOCK, 0, stream>>>(d_c, d_a, d_b, N);
    cudaStreamEndCapture(stream, &graph);

    // 实例化: 编译 graph 为可执行对象
    printf("=== Graph: 实例化 ===\n");
    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

    // 启动: 提交 graph（单次 vs 多次）
    printf("=== Graph: 单次启动 ===\n");
    cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);

    printf("=== Graph: 多次启动（重用） ===\n");
    for (int i = 0; i < 3; i++) {
        cudaGraphLaunch(graph_exec, stream);
    }
    cudaStreamSynchronize(stream);

    // 清理
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a;
    printf("OK\n");
    return 0;
}

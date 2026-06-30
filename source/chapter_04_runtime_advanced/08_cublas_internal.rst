cuBLAS 库调用分析：封闭源库的 kernel dispatch 路径
=======================================================

   cuBLAS (CUDA Basic Linear Algebra Subprograms) 是 NVIDIA 的闭源
   线性代数库。与用户手写 kernel 不同，cuBLAS 的调用链多了一层：
   应用 → ``libcublas.so`` → ``libcuda.so`` → 内核。本节通过 strace
   和运行时对比，揭示 cuBLAS 内部的 kernel 选择机制与 launch 模式。

   :doc:`../chapter_03_runtime/03_kernel_launch` 拆解了 ``<<<>>>`` 到
   ``ioctl(0x4e)`` 的用户 launch 路径。cuBLAS 最终同样经 ``cuLaunchKernel``
   提交命令，但 kernel 来自闭源库内预编译的 cubin/PTX 变体，而非用户
   fatbin 中的 ``vector_add``。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 / RTX 4060 Laptop GPU

   测试程序: ``examples/cublas_demo.cu`` (1K×1K SGEMM)

--------------

cuBLAS 库架构
----------------

.. list-table:: cuBLAS 相关库文件
   :header-rows: 1
   :widths: 25 15 60

   * - 文件
     - 大小
     - 角色
   * - ``libcublas.so.13``
     - **52 MB**
     - cuBLAS 高级 API (\ ``cublasSgemm``\ 等)
   * - ``libcublasLt.so.13``
     - **480 MB**
     - cuBLAS 轻量级 API + 所有编译好的 kernel
   * - ``libcuda.so``
     - ~20 MB
     - Driver API (最终调用)

**核心发现：libcublasLt 的 480 MB 中几乎全是预编译的 GPU kernel**
（SASS + PTX）。cuBLAS 在首次调用 ``cublasSgemm`` 时，内部会遍历
这些预编译 kernel，根据输入矩阵特征（M/N/K、layout、数据类型）选择
最优的 kernel launch 配置。

strace 显示 cuBLAS 的加载过程：

.. code:: text

   openat("/usr/local/cuda/lib64/libcublas.so.13")       = 18  ← API 层
   openat("/usr/local/cuda/lib64/libcublasLt.so.13")     = 19  ← 数千个 kernel
   ; cuBLAS 初始化阶段 (cublasCreate):
   ioctl(0x2a) × 3     ← 分配 workspace + 内部结构
   ioctl(0x2b) × 3     ← fence
   ; 无额外的 mmap — cuBLAS 使用 cudaMalloc 分配 workspace

--------------

cuBLAS SGEMM 调用链
----------------------

``cublasSgemm`` 的内部 dispatch 路径：

.. code:: text

   cublasSgemm(handle, ...)
     ├── 1. 验证参数 (M/N/K/alpha/beta/pointer 合法性)
     ├── 2. 查询 kernel cache (是否已有匹配的 kernel 选择)
     ├── 3. kernel selection 启发式规则:
     │     ├── M < 16 → 使用 scalar kernel
     │     ├── K % 4 != 0 → 使用非对齐 kernel
     │     ├── 一般情况 → 使用 tile-based kernel (16×16, 32×32, etc.)
     │     ├── FP16 input + sm_75+ → Tensor Core kernel
     │     └── 否 → 回退到 SIMT kernel
     ├── 4. cuLaunchKernel (1 次或多次 ioctl)
     └── 5. 更新 kernel cache

关键点：**cuBLAS 不通过 ``execve`` 调用 nvcc**。所有 kernel 已在
libcublasLt.so 中预编译为 cubin，运行时只是按需加载和调用。

--------------

性能对比：手写 vs cuBLAS
---------------------------

.. list-table:: 1024×1024 SGEMM (10 次 launch)
   :header-rows: 1
   :widths: 25 20 20 35

   * - 实现
     - 总耗时
     - 平均单次
     - 说明
   * - 手写 naive GEMM
     - 27.12 ms
     - 2.71 ms
     - 全局内存、无 tiling、1 block/thread
   * - cuBLAS SGEMM
     - **3.03 ms**
     - **0.30 ms**
     - 寄存器 tiling、shared mem、同时使用

cuBLAS 优势：

- **计算效率**：cuBLAS 使用寄存器分块（16×16 tile）+ shared memory
  tiling + 向量化加载（``LDG.128``），naive kernel 只用了全局内存
- **kernel 选择**：cuBLAS 内部根据 M/N/K 选择最优 tile 大小和 launch
  配置，手写代码需要手动调参
- **cuBLAS 9× 快于 naive 实现** — 这里的 naive 是最简单的教科书版本，
  使用 shared memory 的优化版本可以达到 cuBLAS 的 60-80% 性能

--------------

ioctl 模式对比
----------------

cuBLAS ``cublasSgemm`` 产生的 ioctl 与手写 kernel 的差异：

.. code:: text

   ; 手写 naive GEMM (1 次 launch)
   0x4e × 1          ← 1 kernel launch
   0x2b × 1          ← 1 fence (synchronize)
   
   ; cuBLAS SGEMM (1 次调用)
   0x4e × 2-3        ← 2-3 个 kernel launch
   0x2b × 2-3        ← 2-3 个 fence
   0x2a × 0          ← 无额外分配 (使用 handle 的 workspace)

**cuBLAS 一个 API 调用内部对应多个 kernel launch**。对于 SGEMM：

1. **scale kernel**：将 C 矩阵乘以 beta（如果 beta != 0）
2. **main GEMM kernel**：实际的矩阵乘（tile-based）
3. **output kernel**：将结果 scale 并写入 C（处理 alpha）

这解释了为什么 cuBLAS ``ioctl(0x4e)`` 计数大于用户态看到的 API 调用
次数——cuBLAS 将单个 ``cublasSgemm`` 拆分为多个 GPU kernel。

--------------

cuBLAS 与 cuBLASLt
---------------------

CUDA 10.1+ 引入了 cuBLASLt (lightweight) API，提供更细粒度的 kernel
选择控制：

.. code:: text

   传统 cuBLAS:
   cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc)
     → 内部自动选择 kernel (黑盒)
   
   cuBLASLt:
   cublasLtMatmulDescCreate(...)     ← 描述矩阵属性
   cublasLtMatmulAlgoGetHeuristic(...)  ← 获取候选算法
   cublasLtMatmul(handle, ...)       ← 使用指定算法
     → 可指定 tile 大小、split-K、SM count 等

cuBLASLt 将 kernel 选择逻辑暴露给用户，支持：

- **Split-K**：将 K 维度拆分到多个 SM 上并行计算
- **Custom tile size**：选择 16/32/64 维度的 tile
- **Workspace size**：控制临时空间大小（影响并行度）

--------------

关键发现
-----------

1. **cuBLAS 的 480 MB 是 kernel 仓库** — ``libcublasLt.so`` 是全书
   中最大的单个二进制文件。它包含数千个为不同 M/N/K、dtype、layout
   组合预编译的 SGEMM kernel 变体。驱动在运行时根据输入特征选择一个。

2. **一个 cuBLAS API = 多个 GPU kernel** — cublasSgemm 在 strace
   层面对应 2-3 个 ioctl(0x4e)（scale → GEMM → output）。这与
   手写 kernel 的 1:1 映射不同。

3. **cuBLAS 不使用 nvcc 子进程** — 与 ``cudaMallocAsync`` 的显存池
   类似，cuBLAS 将所有 kernel 预编译打包在 ``.so`` 中，运行时只需
   按需加载。这与 ``cuModuleLoadData`` 加载外部 cubin 的方式一致。

4. **cuBLAS 的性能来自 tens of thousands of engineer-hours** —
   libcublasLt 的 480 MB 中有大量手工调优的 SASS 和 PTX 变体。
   "cuBLAS 很快"不是因为运行时做了神奇的事，而是因为所有可能的
   输入组合都提前被调优好了。

5. **cuBLAS 不是一个"编译器"** — 虽然它"选择 kernel"，但这个选择
   是查表（heuristic + empirical profiling），而非实时编译。
   这与 NVRTC 或 PTX JIT 的动态编译有本质区别。

*分析基于 CUDA 13.1 (cuBLAS 13.2.1.1) / RTX 4060 Laptop GPU。naive GEMM*
*仅为教学用途，实际应用中应使用 cuBLAS、CUTLASS 或 Triton。*

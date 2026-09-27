NVRTC 在线编译：进程内编译 vs nvcc 子进程
=============================================

   NVRTC (NVIDIA Runtime Compilation) 允许在程序运行时动态编译
   CUDA kernel 源码。与 :doc:`../chapter_01_compilation/01_compilation_pipeline`
   分析的 nvcc 离线编译不同，NVRTC **不创建子进程**——整个 LLVM
   编译管道在调用进程内完成。本节用 strace 对比两种路径的系统
   调用差异。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 / RTX 4060 Laptop GPU

   测试程序: ``examples/nvrtc_demo.cu`` (4M vec_add)

--------------

两种编译路径
--------------

.. code:: text

   离线编译 (nvcc):
   .cu → nvcc (exec cicc → ptxas → fatbinary) → .o → g++ → executable
   |_____ 构建时完成 _____|                    |_链接_|

   在线编译 (NVRTC):
   .cu string → nvrtcCompileProgram (in-process LLVM) → PTX → cuModuleLoadData
   |_______________ 运行时完成 ________________|

   两个路径的差别类似 GCC 和 LLVM JIT 的区别——前者产生持久化文件，
   后者在内存中完成所有编译。

两条编译路径的完整对比：

.. mermaid:: ../_static/nvrtc_flow.mmd

--------------

系统调用对比
---------------

strace 捕获的核心差异：

.. code:: text

   ; 离线编译 (nvcc, 构建时) — tracker 统计
   execve("cicc")     = 1  ← CUDA 前端 + PTX 生成
   execve("ptxas")    = 1  ← PTX → SASS
   execve("fatbinary") = 1 ← 打包 cubin
   ; 总计: ~12 次 execve (含 g++, nvlink, ld 等)
   
   ; 在线编译 (NVRTC, 运行时) — 当前进程 strace
   execve(...)        = 0  ← **无子进程**
   openat("libnvrtc.so") = 16 ← 加载 NVRTC 编译器库
   openat("libnvrtc-builtins.so") = 17 ← 内置数学函数
   ; 所有编译在调用线程内完成

关键数据：

.. list-table:: 两种路径的关键指标
   :header-rows: 1
   :widths: 25 25 25 25

   * - 指标
     - 离线 (nvcc)
     - 在线 (NVRTC)
     - 差异
   * - execve 次数
     - ~12
     - **0**
     - NVRTC 零子进程
   * - 编译时间
     - 0 ms (预编译)
     - **254 ms**
     - nvcc 提前完成
   * - PTX 大小
     - 嵌入 fatbin
     - 1117 bytes
     - NVRTC 可直接获取
   * - Launch 时间
     - 0.285 ms
     - 0.302 ms
     - 几乎相同
   * - 总计 (编译+launch)
     - 0.285 ms
     - 254.57 ms
     - nvcc 快 892×

两种路径在 launch 阶段几乎完全相同（0.28 vs 0.30 ms），差异
全部来自编译阶段。

--------------

NVRTC 编译在什么阶段产生开销？
----------------------------------

通过 ``libnvrtc.so`` 的分析，NVRTC 编译分为三个阶段：

.. code:: text

   nvrtcCompileProgram 内部流程:
   ├── 1. 语法分析 (EDG 前端, 嵌入在 libnvrtc.so)
   ├── 2. NVVM IR 生成 (LLVM bitcode)
   ├── 3. NVVM → PTX (LLVM 后端 → PTX 后端)
   └── 输出: PTX 文本

   ; 注意: NVRTC 不调用 ptxas——用户需要自行调用
   ; cuModuleLoadData 会触发驱动 JIT 将 PTX 编译为 SASS

NVRTC 只输出 PTX，不输出 cubin。这与 nvcc 的完整流水线
（cicc → ptxas → fatbinary）不同。在 NVRTC 路径中，
**PTX → SASS 的编译由驱动在 ``cuModuleLoadData`` 时完成**。

这意味着：

- NVRTC 的 ``nvrtcCompileProgram`` 只负责 **源码 → PTX**
- PTX 到 SASS 的 JIT 编译是**驱动内部行为**，由 ``cuModuleLoadData`` 触发
- 如果多次加载相同的 PTX，驱动会缓存编译结果（模块缓存）

--------------

何时使用 NVRTC？
------------------

.. list-table:: NVRTC 适用场景
   :header-rows: 1
   :widths: 30 70

   * - 场景
     - 说明
   * - **JIT 特殊化**
     - 用户输入决定 kernel 参数（如滤波器大小），编译时嵌入常量
       可让编译器做更好的优化
   * - **GPU 架构自适应**
     - 运行时检测 GPU 架构（sm_75/sm_89/sm_100），编译针对性的 kernel
   * - **脚本语言绑定**
     - Python/Julia 等语言的 CUDA 扩展在运行时生成 PTX
   * - **避免 nvcc 依赖**
     - 容器化部署中可能没有完整 CUDA Toolkit，只有 libnvrtc
   * - **AOT 编译**
     - 提前编译 PTX 并缓存到磁盘，后续运行直接加载

不适合的：

- **固定 kernel** — 如果 kernel 代码在编译时已知，nvcc 更优
- **短生命周期进程** — 编译 254 ms 可能超过程序总运行时间
- **循环中编译** — 应缓存编译结果

--------------

NVRTC 与 cuModuleLoadData 的协作
------------------------------------

NVRTC 最自然的使用方式是与 Driver API 配合：

.. code:: text

   1. nvrtcCreateProgram(source)
   2. nvrtcCompileProgram(...)     → PTX (1117 bytes)
   3. nvrtcGetPTX(ptx_buffer)
   4. cuModuleLoadData(ptx)        → 驱动 JIT PTX→SASS + 加载
   5. cuModuleGetFunction(...)     → 获取 kernel 句柄
   6. cuLaunchKernel(...)          → 提交到 GPU

这与常规 Runtime API 的 ``<<<>>>`` 路径不同——NVRTC 强制使用
Driver API（因为 kernel 是动态生成的，没有编译时的 stub 函数）。

strace 显示 Driver API 路径的 ioctl 与 Runtime API 路径相同：
``ioctl(0x4e)`` 代表 kernel launch，``ioctl(0x2b)`` 代表同步。
差异仅在前端——NVRTC 的编译发生在调用进程内，nvcc 的编译发生在
独立子进程中。

--------------

关键发现
-----------

1. **NVRTC 零子进程** — strace 确认 ``nvrtcCompileProgram`` 不产生
   execve。libnvrtc.so 将所有 LLVM 编译管道嵌入为一个 97 MB 的
   共享库，编译完全在调用进程内完成。

2. **NVRTC 不产生 ptxas 子进程** — 这是与 nvcc 路径最基本的区别。
   NVRTC 只输出 PTX，PTX→SASS 的编译推迟到驱动层（cuModuleLoadData
   JIT）。这意味着 NVRTC 路径的**最终代码质量**取决于驱动 JIT 的
   优化能力——通常与 ptxas -O3 相当，但没有 ptxas 的寄存器分配 tuning。

3. **编译开销 254 ms** — 对于简单的 vec_add，NVRTC 编译耗时 254 ms。
   这个开销来自 LLVM 管道（IR 生成、优化 pass、PTX 后端）。kernel
   越复杂，编译时间增长越快。实际应用中，应编译一次、缓存 PTX、
   多次 launch。

4. **Launch 无差异** — PTX 被 driver JIT 为 SASS 后，kernel launch
   的 ioctl 行为与预编译 cubin 完全相同。编译路径的差异不影响
   GPU 执行效率。

5. **NVRTC 是连接编译时与运行时的桥梁** — 沿用了 :doc:`03_module_loading`
   分析的 ``cuModuleLoadData`` 加载路径，但 PTX 由进程内 LLVM
   生成而非从文件系统读取。这为 CUDA 的 JIT 生态（如 Python
   Numba、SYCL 的 PTX 后端）提供了基础。

*分析基于 CUDA 13.1 / Driver 595.58.03 / libnvrtc 13.1 / RTX 4060 Laptop GPU。
NVRTC 编译时间因 kernel 复杂度而异，vec_add 为 254 ms，复杂模板可达数秒。*

*Deep Dive Into CUDA — 2026 年 6 月*

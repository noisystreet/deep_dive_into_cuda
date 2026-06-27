Streams 与 CUDA Graphs 深度分析
======================================

   Streams 和 CUDA Graphs 是 CUDA 并发与编译的两种核心抽象。本节用
   strace 和 ioctl 模式分析它们的 **运行时真实行为**：stream 如何提交到

.. admonition:: 你知道吗？

   你可能在 PyTorch/TensorFlow 中使用过 CUDA Streams 而不自知——
   每次调用 ``torch.cuda.synchronize()`` 背后就是一个 stream 同步
   操作。深度学习框架的默认行为是：每个 CUDA 设备有一个默认 stream，
   所有操作（前向、反向、优化）都在这个 stream 上串行执行。这也是
   为什么 ``torch.cuda.Stream`` 可以加速推理——通过将不同层的计算
   分配到不同 stream 实现**计算和传输的重叠**。你可以在 PyTorch 中
   用 ``with torch.cuda.stream(stream):`` 将操作指派到特定 stream。

   GPU 命令队列，Graph 实例化是否走 nvcc 子进程，以及它们与普通
   launch 的系统调用差异。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) / Linux x86-64

   测试程序: ``examples/streams_demo.cu``, ``examples/graph_demo.cu``

--------------

Stream 的本质
---------------

``cudaStreamCreate`` 创建了什么？从系统调用层面看：

::

   $ strace -e ioctl,openat ./streams_demo

在 ``cudaStreamCreate`` 调用期间，**没有产生任何 ioctl**。stream 完全
是用户态对象——``libcudart`` 分配一个 ``cudaStream_t`` 整数 ID，driver
在内部维护 stream 对应的 GPU 命令队列。

**结论**：Stream 不是内核对象，不涉及系统调用。它是一个**逻辑命令队列标
识符**，所有 stream 共享同一个 GPU 硬件队列，通过编码在命令缓冲区中的
stream ID 实现逻辑隔离。

.. mermaid:: ../_static/streams_flow.mmd

默认 stream 的阻塞语义
~~~~~~~~~~~~~~~~~~~~~~~

默认 stream (``NULL``) 是**同步**的。当一个非阻塞 stream 有任务在执
行时，默认 stream 上的 launch 会 **隐式插入一个 fence**——driver 会先发
送一个 ``ioctl(0x2b)`` 等待之前所有 stream 完成，再提交新的任务。

strace 证据：``streams_demo`` 的最后一部分（默认 stream 与 s1 混合）比
纯非阻塞 stream 多 1 次 ``ioctl(0x2b)``：

::

   ; 纯非阻塞
   ioctl(0x4e) × 2   ← s1: vec_add, s2: vec_mul
   ioctl(0x2b) × 2   ← cudaStreamSynchronize(s1), cudaStreamSynchronize(s2)

   ; 默认 stream + 非阻塞（阻塞语义）
   ioctl(0x2b) × 1   ← 隐式 fence: 等待 s1 完成
   ioctl(0x4e) × 1   ← 默认 stream: vec_mul
   ioctl(0x2b) × 1   ← cudaDeviceSynchronize

Event 同步：无 ioctl 的 GPU 栅栏
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaEventRecord`` 和 ``cudaStreamWaitEvent`` **不产生任何系统调用**。
Event 是一个 GPU 侧的标记点：

1. ``cudaEventRecord(e, s)`` — 在 stream s 的命令队列中插入一个 "记录时
   间戳" 标记。GPU 执行到该标记时写入事件时间戳。
2. ``cudaStreamWaitEvent(s2, e)`` — 在 stream s2 的命令队列中插入一个
   "等待事件 e 完成" 的栅栏。GPU 在 s2 上遇到此栅栏时会暂停，直到 s1
   执行到 ``cudaEventRecord`` 标记后才继续。

这个过程完全在 **GPU 命令缓冲区层面** 完成，driver 只在最终提交命令时
通过 ``ioctl(0x4e)`` 将缓冲区发送到 GPU，没有单独的 ioctl 用于
Event 同步。

--------------

CUDA Graphs 深度追踪
-----------------------

CUDA Graphs 提供了一种将多个 kernel launch 打包为单个计算图的机制。
本节重点分析其**实例化**和**启动**两个阶段的系统调用特征。

.. mermaid:: ../_static/graph_sequence.mmd

捕获阶段：零额外开销
~~~~~~~~~~~~~~~~~~~~~

``cudaStreamBeginCapture`` 到 ``cudaStreamEndCapture`` 之间，所有 kernel
launch **不会实际提交到 GPU**，而是被 driver 记录为图节点：

::

   cudaStreamBeginCapture(stream)
   vec_add<<<>>> (stream)    ← 记录节点 A
   vec_mul<<<>>> (stream)    ← 记录节点 B
   cudaStreamEndCapture(stream, &graph)

此阶段 **无 ioctl**。strace 显示：

::

   ; 在此时间窗口内没有任何 ioctl(0x4e) 调用

实例化阶段：进程内 JIT
~~~~~~~~~~~~~~~~~~~~~~~

``cudaGraphInstantiate(&graph_exec, graph, ...)`` 将捕获的图编译为可执
行对象。关键发现：**不产生 execve 子进程**。

::

   $ strace -f -e execve,clone3 ./graph_demo 2>&1 | grep -E 'ptxas|cicc|nvlink'
   ; 无输出！Graph 实例化不走 nvcc 子进程

strace 也确认没有 ``clone3``（除 GPU worker 线程外），没有 ``execve``。
Graph 编译完全在 **libcuda 进程内部** 完成——driver 使用内置的 NVVM JIT
引擎（与 PTX JIT 回退同一套）直接将 graph 降为 SASS，无需调用 cicc 或
ptxas 子进程。

这意味着 CUDA 有三套并存的编译路径：

.. list-table:: CUDA 编译路径对比
   :header-rows: 1
   :widths: 15 25 25 35

   * - 路径
     - 入口
     - JIT 方式
     - ioctl/execve
   * - **离线编译**
     - ``nvcc`` (命令行)
     - execve 子进程
       (cicc → ptxas)
     - 多次 execve
   * - **PTX JIT**
     - 首次加载 fat binary
     - 进程内 NVVM
       (libcuda dlopen)
     - 无 execve
   * - **Graph JIT**
     - ``cudaGraphInstantiate``
     - 进程内 NVVM
       (libcuda dlopen)
     - 无 execve

启动阶段：与普通 launch 相同的 ioctl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaGraphLaunch(graph_exec, stream)`` 提交 graph 执行。从 strace 视
角看，它与普通 kernel launch 没有本质区别：

::

   ; 普通 launch
   ioctl(0x4e) × 1   ← cuLaunchKernel

   ; Graph launch
   ioctl(0x4e) × 1   ← cuGraphLaunch

每次 ``cudaGraphLaunch`` 都产生一次 ``ioctl(0x4e)``。**Graph 不减少
硬件操作**——它减少的是 **Driver API 调用次数和参数校验开销**。对于普通
launch，每调用一次 ``cuLaunchKernel`` 就要构建一次命令缓冲区；对于
Graph，命令缓冲区在实例化阶段预先构建好，launch 时直接提交。

多次启动（Graph 重用）
~~~~~~~~~~~~~~~~~~~~~~~

:: code::

   for (int i = 0; i < 3; i++) {
       cudaGraphLaunch(graph_exec, stream);
   }
   cudaStreamSynchronize(stream);

strace 显示 3 次 ``ioctl(0x4e)`` + 1 次 ``ioctl(0x2b)``。Graph 的可重
用性避免了多次 launch 场景下反复的参数校验和命令构建开销，但在系统调用
层面，每次 graph launch 仍然是独立的 ``ioctl(0x4e)``。

--------------

ioctl 统计摘要
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 25 10

   * - 程序
     - ioctl 总数
     - ``0x4e`` 次数
     - ``0x2b`` 次数
     - 说明
   * - ``vector_add``
     - 104
     - 25
     - 0
     - 单 kernel launch
   * - ``streams_demo``
     - 161
     - 25
     - 103
     - 多 stream + event
   * - ``graph_demo``
     - 161
     - 25
     - 102
     - Graph 捕获 + 1+3 次 launch

三个程序的核心 kernel launch 次数不同（vector_add ×1, streams_demo ×5, graph_demo ×6），但 ``0x4e`` 次数相近（25），这是因为大部分 ``0x4e`` 用于 kernel launch 之外的 DMA 控制（memcpy 内部）。

关键发现
-----------

1. **Stream 是用户态对象** — ``cudaStreamCreate`` 无 ioctl，stream ID
   编码在命令缓冲区数据中，driver 提交时一并发送。

2. **Event 是 GPU 栅栏** — ``cudaEventRecord`` 和
   ``cudaStreamWaitEvent`` 无 ioctl，由 GPU 命令队列中的标记和等待
   指令实现，硬件级同步。

3. **默认 stream 的隐式 fence** — 默认 stream 与非阻塞 stream 混合使
   用时会触发额外的 ``ioctl(0x2b)``，这是 driver 的同步保障。

4. **Graph 实例化不走 nvcc** — ``cudaGraphInstantiate`` 的 JIT 编译
   在 libcuda 进程内完成（NVVM 引擎），不产生 execve 子进程。

5. **Graph launch = 普通 launch** — 系统调用层面，``cudaGraphLaunch``
   与 ``cuLaunchKernel`` 都通过 ``ioctl(0x4e)`` 提交，Graph 的优势
   在用户态（减少参数校验和命令构建），而非减少 ioctl。

6. **三套编译路径** — CUDA 的离线编译、PTX JIT、Graph JIT 使用完全不
   同的进程模型：execve 子进程 vs 进程内 dlopen。这点在理解 CUDA 编译
   架构时至关重要。

*分析基于 CUDA 13.1 / Driver 595.58.03。strace 统计含初始化阶段的 ioctl。*

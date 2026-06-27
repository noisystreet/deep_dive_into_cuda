同步机制深度分析
==================

   聚焦 ``vector_add.cu`` 第 41 行的 ``cudaDeviceSynchronize()``\ ，用
   strace 还原 Runtime → Driver → 内核 → eventfd/futex 的完整等待路径。

.. admonition:: 你知道吗？

   NVIDIA 的 CPU-GPU 同步模型有一个鲜为人知的事实：
   ``cudaDeviceSynchronize`` 不是发送一个新命令，而是等待一个
   已经提交的 fence。每次 ``cuLaunchKernel`` 都会**附带提交一个
   fence**——``cudaDeviceSynchronize`` 就是阻塞 CPU 直到那个 fence
   完成。这解释了为什么连续调用两次同步几乎没有额外开销。
   真正昂贵的不是同步本身，而是 GPU 执行所有排队工作所需的时间。


环境: CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) / Linux x86-64

--------------

为什么需要同步？
----------------

:doc:`03_kernel_launch` 分析了 kernel 如何通过 ``ioctl(0x4e)`` 提交到
GPU。但 launch 调用 **立即返回**——CPU 不会等待 GPU 算完。``vector_add`` 在
launch 后立刻调用 ``cudaDeviceSynchronize()``\ ，确保后续 ``cudaMemcpy(D2H)``
读到的是完整结果。

.. code:: cuda

   vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
   cudaDeviceSynchronize();                    // ← 本节焦点
   cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

若去掉同步，D2H 拷贝可能与 kernel 并发，读到未写完的数据。

CUDA 提供三级同步 API：

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - API
     - 粒度
     - Driver 对应
   * - ``cudaDeviceSynchronize()``
     - 整个 GPU 设备
     - ``cuCtxSynchronize_v2()``
   * - ``cudaStreamSynchronize(stream)``
     - 单个 stream
     - ``cuStreamSynchronize_ptsz()``
   * - ``cudaEventSynchronize(event)``
     - 单个 event
     - ``cuEventSynchronize()``

本节以 ``cudaDeviceSynchronize`` 为主，其他 API 走相同的底层机制（fence +
eventfd + futex），仅等待范围不同。

--------------

分析工具
--------

=================== ========================================
工具                用途
=================== ========================================
``strace -f -tt``   跟踪主线程 + worker 线程，带时间戳
``ioctl`` 过滤      统计 ``0x2b`` (fence) 出现次数与模式
``futex`` 过滤      观察 WAIT/WAKE 配对
``objdump -d``      反汇编 ``cudaDeviceSynchronize``
``nm -C``           确认 Runtime → Driver 符号映射
=================== ========================================

复现命令：

.. code:: bash

   cd examples/build
   strace -f -tt -e trace=ioctl,eventfd2,clone3,futex,poll,read,write \
     ./vector_add 2>&1 | tee sync_trace.txt

--------------

Runtime 层：cudaDeviceSynchronize
------------------------------------

``nm -C vector_add`` 显示同步相关符号：

.. code:: text

   cudaDeviceSynchronize          ← Runtime API（嵌入 libcudart_static.a）
   cudaStreamSynchronize
   cudaEventSynchronize

``libcuda.so.1`` 导出：

.. code:: text

   cuCtxSynchronize
   cuCtxSynchronize_v2            ← Runtime 实际调用的版本
   cuStreamSynchronize / _ptsz
   cuEventSynchronize

调用链：

.. code:: text

   cudaDeviceSynchronize()          // vector_add.cu:41
     → cuCtxSynchronize_v2()        // libcuda.so
       → ioctl(/dev/nvidia0, 0x2b)  // fence 查询
       → futex(WAIT) / eventfd      // 阻塞等待 GPU 完成

``cudaDeviceSynchronize`` 函数体约 380 字节（``objdump``\ ），逻辑为：获取当
前 CUDA 上下文 → 校验有效性 → 排空所有 stream → 轮询 fence 直到 GPU 空闲。

--------------

ioctl(0x2b)：Fence 同步命令
---------------------------

:doc:`01_driver_interface` 已将 ``ioctl(NR=0x2b)`` 归类为\ **同步 / 等待 /
Barrier**\ 。对 ``vector_add`` 完整运行的统计：

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - ioctl 命令
     - 次数
     - 功能
   * - ``0x2a`` (NR=42)
     - 161
     - 内存管理
   * - ``0x2b`` (NR=43)
     - **102**
     - Fence 同步 / 等待 GPU
   * - ``0x4e`` (NR=78)
     - 25
     - GPU 命令提交 (DMA + launch)
   * - ``0xc9`` (NR=201)
     - 10
     - 上下文创建

命令格式：

.. code:: text

   ioctl(fd=10, _IOC(R|W, type=0x46, nr=0x2b, size=0x30), argp)
   // type='F', 数据大小 48 字节

``0x2b`` 占全部 410 次 device ioctl 的 **25%**\ ，是仅次于 ``0x2a`` 的第
二高频命令——说明同步开销在简单程序中已相当显著。

102 次 ``0x2b`` 并非全部来自 ``cudaDeviceSynchronize``\ 。按出现时机划分：

.. code:: text

   launch 之前:     7 次   ← 初始化 / 上下文准备
   launch 期间:    85 次   ← 与 0x4e (DMA/launch) 交织的 fence 轮询
   launch 之后:    10 次   ← cudaDeviceSynchronize + 清理

launch 期间的大量 ``0x2b`` 说明：即使不显式调用同步 API，``cudaMemcpy`` 和
``cudaLaunchKernel`` 内部也会 **隐式 fence**——确保前序 GPU 操作完成后再
提交新命令。

--------------

cudaDeviceSynchronize 的 strace 时间线
-----------------------------------------

kernel launch 的最后一次 ``ioctl(0x4e)`` 出现在 strace 第 466 行
（``22:27:51.819757``\ ）。此后 ``cudaDeviceSynchronize`` 的等待过程分为
两个阶段：

阶段 1 — 排空 Stream（约 51.821–51.823 s）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

最后一条 ``0x4e`` 之后，主线程对每个活跃 stream 执行一轮同步：

.. code:: text

   475  ioctl(25, 0xc9)              ← 上下文操作
   476  ioctl(25, 0xce)              ← 事件管理
   477  ioctl(25, 0x2b)              ← fence 查询
   478  write(17, "\1\0\0\0\0\0\0\0", 8)   ← 向 eventfd 写入完成信号
   479  futex(WAIT_BITSET)           ← 主线程阻塞
   480  [worker 266643] poll(fd=17)  ← worker 检测到 eventfd 可读
   481  [worker 266643] read(17, 8) ← 读取 8 字节
   482  [worker 266643] futex(WAKE)  ← 唤醒主线程
   483  futex resumed                ← 主线程恢复

``write(17, ... 8 字节)`` 中的 fd 17 来自 ``eventfd2(EFD_CLOEXEC|EFD_NONBLOCK)``
（strace 第 756ms 处创建）。8 字节值 ``\1\0\0\0\0\0\0\0`` 是 eventfd 的标准
计数信号。

此模式对 fd 25/27/29/31（多个 stream 通道）**重复 4 次**\ ，每次约 0.5 ms。

阶段 2 — 最终 Fence 确认（约 51.826 s）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stream 排空后，主线程对 ``/dev/nvidia0`` (fd=10) 连续 3 次 fence 查询：

.. code:: text

   547  ioctl(10, 0x2b, 0x30)   ← fence #1
   550  ioctl(10, 0x2b, 0x30)   ← fence #2
   553  ioctl(10, 0x2b, 0x30)   ← fence #3（确认 GPU 完全空闲）

三次轮询间隔约 0.05–0.07 ms，是典型的 fence 寄存器轮询模式——驱动反复
查询 GPU 硬件 fence 值，直到所有已提交的命令完成。

从最后一条 ``0x4e`` (``.819757``) 到最终 ``0x2b`` (``.826282``)\ ，
``cudaDeviceSynchronize`` 耗时约 6.5 ms（含 4096 block 的 kernel 执行时间）。

.. mermaid:: ../_static/sync_mechanism.mmd

--------------

Worker 线程与 eventfd 机制
---------------------------

CUDA 运行时在首次 GPU 操作时创建 4 个 worker 线程（:doc:`01_driver_interface`
Phase 6）：

.. code:: text

   eventfd2(0, EFD_CLOEXEC|EFD_NONBLOCK) = 5    ← 第 1 个 eventfd
   clone3(...) = 266638                          ← worker 线程 1
   ...
   eventfd2(...) = 35                            ← 第 4 个 eventfd
   clone3(...) = 266644                          ← worker 线程 4

Worker 线程职责：

1. ``poll()`` 监听多个 eventfd 文件描述符
2. GPU 完成时读取 eventfd（``read(fd, 8)``）
3. ``futex(WAKE)`` 唤醒在 ``futex(WAIT)`` 上阻塞的主线程

完整通知链：

.. code:: text

   GPU 执行完成
     → MSI 硬件中断
     → nvidia.ko IRQ handler
     → 更新 eventfd 计数器
     → worker: poll() 返回 POLLIN
     → worker: read(eventfd, 8)
     → worker: futex(WAKE, main_thread)
     → main: futex(WAIT) 返回，继续执行

这种设计将 **中断处理** 放在内核和 worker 线程中，主线程仅通过 futex 阻塞/
唤醒，避免 busy-wait 占用 CPU。

strace 中 futex 统计：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 操作
     - 含义
   * - ``FUTEX_WAIT_BITSET_PRIVATE`` (主线程)
     - 带超时的阻塞等待（``tv_sec`` + ``tv_nsec``）
   * - ``FUTEX_WAKE_PRIVATE`` (worker)
     - 唤醒 1 个等待线程
   * - 总计 50 次 futex 调用
     - 含初始化 WAKE 和同步 WAIT/WAKE 配对

--------------

隐式同步 vs 显式同步
--------------------

``vector_add`` 中有两类同步需求：

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 场景
     - 触发方式
     - strace 表现
   * - ``cudaMemcpy(H2D)`` 前
     - 隐式：确保目标 buffer 就绪
     - ``0x2b`` 与 ``0x4e`` 交织
   * - ``cudaLaunchKernel`` 后
     - 隐式：默认 stream 排队
     - launch 期间 85 次 ``0x2b``
   * - ``cudaDeviceSynchronize()``
     - 显式：等待全部 GPU 工作
     - stream 排空 + 3 次最终 fence
   * - ``cudaMemcpy(D2H)`` 前
     - 依赖上面的显式同步
     - 若去掉 Sync，此处可能读到脏数据

默认 stream（stream 0）上的操作是 **顺序执行** 的——Runtime 在每次提交
``0x4e`` 命令前会插入 ``0x2b`` fence 确保前序操作完成。但 **跨 stream**
或 **异步操作** 必须显式同步。

--------------

三种同步 API 的对比
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - 特性
     - DeviceSynchronize
     - StreamSynchronize
     - EventSynchronize
   * - 等待范围
     - 所有 stream
     - 单个 stream
     - 单个 event 记录点
   * - 典型用途
     - 程序级 barrier
     - 流水线 stage 同步
     - 精确计时 / 依赖
   * - 底层 ioctl
     - ``0x2b`` × N streams + 最终 fence
     - ``0x2b`` × 1 stream
     - ``0x2b`` + event 查询
   * - 阻塞主线程
     - 是
     - 是
     - 是

三者共享 eventfd + futex 等待机制，区别仅在于 Driver 层等待哪些 GPU 队列
清空。

--------------

与 Kernel Launch 的衔接
------------------------

将 :doc:`03_kernel_launch` 和本节串联，``vector_add`` 的 GPU 操作时间线：

.. code:: text

   51.706 s   cudaMemcpy(H2D) × 2        ← 0x4e + 0x2b 交织
   51.819 s   cudaLaunchKernel           ← 最后一条 0x4e
   51.821 s   排空 stream (×4)           ← write(eventfd) + futex
   51.826 s   cudaDeviceSynchronize      ← 最终 0x2b × 3
   51.831 s   cudaMemcpy(D2H)            ← 安全读取结果

Kernel 提交（``0x4e``）与完成确认（``0x2b``）之间相隔约 **6.5 ms**——这
是 4096 block × 256 thread 向量加法在 sm_89 上的实际 GPU 执行 + 同步开销。

--------------

关键发现
--------

1. ``ioctl(0x2b)`` 出现 102 次，占 device ioctl 的 25%——同步并非"一次调用
   一次 ioctl"，而是大量 fence 轮询。

2. ``cudaDeviceSynchronize`` 分两阶段：先逐 stream 排空（eventfd + futex × 4），
   再三重 fence 确认 GPU 全局空闲。

3. eventfd 8 字节写入 + futex WAIT/WAKE 是 GPU→CPU 完成通知的标准路径，worker
   线程负责从中断到用户态的桥接。

4. launch 期间 85 次 ``0x2b`` 表明 ``cudaMemcpy`` / ``cudaLaunchKernel`` 内部
   有隐式 fence——不调用 ``cudaDeviceSynchronize`` 也不能保证 D2H 安全。

5. 从最后 ``0x4e`` 到最终 ``0x2b`` 约 6.5 ms，其中包含 kernel 执行时间与
   fence 轮询开销。

6. 4 个 worker 线程 + 4 个 eventfd 在首次 GPU 操作时创建，程序生命周期内复用。

--------------

*Deep Dive Into CUDA — 2026 年 6 月*

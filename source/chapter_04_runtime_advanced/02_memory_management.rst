CUDA 内存管理深度分析
=================================

   CUDA 程序能同时访问 CPU 内存和 GPU 显存，这种双重内存模型是其
   编程复杂性的核心来源。本节通过 strace 和 ``/proc/pid/maps`` 捕获

.. admonition:: 你知道吗？

   CUDA 中有一个常用的性能优化技巧你可能不知道：
   ``cudaHostAlloc`` 分配的 pinned memory 不仅对 GPU→CPU 传输
   速度有巨大影响（5-10 倍于 pageable memory），还会改变 CPU 端的
   内存行为。因为物理页被锁定，系统无法将这些页换出到磁盘——
   过度分配 pinned memory 会导致系统可用物理内存减少，甚至触发
   OOM。这就是为什么生产环境通常**限制 pinned memory 使用量**。
   一个常见的做法是用 ``cudaHostRegister`` 将已分配的 malloc
   内存注册为 pinned，用完后注销，而非全部使用 ``cudaHostAlloc``。

   每种分配路径的实际系统调用，揭示 ``cudaMalloc``、``cudaHostAlloc``、
   ``cudaMallocManaged`` 和 ``cudaMallocAsync`` 在内核侧的真实差异。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) / Linux x86-64

   测试程序: ``examples/memory_demo.cu`` (覆盖 6 种分配场景)

--------------

GPU BAR 与物理地址映射
------------------------

GPU 通过 PCIe BAR (Base Address Register) 将显存暴露给 CPU。
``/proc/pid/maps`` 可以直观看到 CUDA 初始化后用户态的地址空间布局：

.. code:: text

   200000000-200200000 ---p                    ← PROT_NONE 保留区域
   200200000-200400000 rw-s /dev/nvidia0       ← GPU BAR 0 (4 MB)
   200400000-203400000 rw-s /dev/nvidiactl     ← GPU BAR 1 (48 MB)
   203400000-206000000 ---p                    ← PROT_NONE 保留
   206000000-206200000 rw-s /dev/nvidiactl     ← BAR 2 (2 MB)
   206200000-206400000 rw-s /dev/nvidia-uvm    ← UVM BAR (2 MB)
   206400000-300200000 ---p                    ← 大块 PROT_NONE 保留

关键点：

- ``/dev/nvidia0`` 是 GPU 设备节点，**直接映射 GPU BAR**——CPU 通过
  mmap 这个文件可以直接读写 GPU 显存（除非被 GPU 自身遮挡）。
- ``/dev/nvidiactl`` 是控制节点，多个映射用于**上下文切换和命令提交**。
- ``/dev/nvidia-uvm`` 是 UVM (Unified Virtual Memory) 驱动模块，见
  第 4 节。
- ``---p`` (PROT_NONE) 区域是 **GPU 预留地址空间**——它们占位但不分配
  物理页面，访问会触发 segment fault。

``cudaMalloc`` 系统调用：三步走
----------------------------------

``cudaMalloc`` 分配 GPU 显存时，strace 显示以下 ioctl 序列：

.. code:: text

   ioctl(fd, NV_DEV_IOCTL(0x2a), ...)    ← 分配 GPU 物理显存
   ioctl(fd, NV_DEV_IOCTL(0x2b), ...)    ← fence / 同步

每次 ``cudaMalloc`` 对应 **1 次 ioctl(0x2a)**，不产生新的 mmap——GPU
虚拟地址已在初始化阶段预映射到 PROT_NONE 区域，``cudaMalloc`` 只是在
GPU 驱动内部建立页表映射。``cudaMemset`` 则通过 ``ioctl(0x4e)``
发起 DMA 写入（触发物理显存的实际分配）。

.. list-table:: ``cudaMalloc`` 不同大小的 ioctl 模式
   :header-rows: 1
   :widths: 20 25 25 30

   * - 大小
     - ioctl(0x2a) 次数
     - mmap 次数
     - 说明
   * - 8 MB
     - 1
     - 0
     - 虚地址已有，只需页表
   * - 256 MB
     - 1
     - 0
     - 同上的大块分配
   * - 1 GB (连续)
     - 多次
     - 0
     - 可能需要分段映射

--------------

``cudaHostAlloc``：Pinned Memory 的 DMA 通道
-----------------------------------------------

默认的 ``malloc`` 分配的是 pageable memory（可分页内存）——物理页面
可以被内核换出，DMA 引擎无法直接访问。``cudaHostAlloc`` 分配
pinned memory（锁定内存），物理页面被锁定在内存中，DMA 引擎可以
直接读写。

strace 差异：

.. code:: text

   ; malloc(8 MB) — pageable
   brk(NULL)            = 0x...               ← 堆扩张
   ; 无 ioctl，无特殊 mmap
   ; 运行时 memcpy 时：内核临时锁定页面（不可见）

   ; cudaHostAlloc(8 MB) — pinned
   mmap(NULL, 8MB, PROT_READ|PROT_WRITE,
        MAP_SHARED, /dev/zero (deleted), 0)   ← 分配物理页
   ioctl(NV_DEV_IOCTL(0x2a), ...)             ← 注册 DMA 地址

``/dev/zero (deleted)`` 文件是 Linux 内核分配匿名大页的机制。多个
``/dev/zero (deleted)`` 映射（图中 ``/dev/zero`` 条目）对应不同
pinned memory 分配。

``cudaHostAlloc`` Mapped 模式有一个独特的行为：**双映射**。
``cudaHostGetDevicePointer`` 返回一个 GPU 侧地址，该地址在 GPU BAR
空间中指向同一物理页面。CPU 写 → GPU 可见，GPU 写 → CPU 可见，
无需 ``cudaMemcpy``。

--------------

``cudaMallocManaged``：UVM Page Fault 路径
----------------------------------------------

UVM (Unified Virtual Memory) 通过 ``nvidia_uvm.ko`` 内核模块实现透明
页面迁移。``/dev/nvidia-uvm`` 的映射（206200000-206400000）是 UVM
驱动的控制通道。

``cudaMallocManaged`` 只在 UVM 驱动中注册虚拟地址，**不分配任何物理
页面**。物理页面在**首次访问**时按需分配：

.. code:: text

   cudaMallocManaged(8 MB)     → ioctl(0x2a) 注册虚地址 (UVM)
   m_ptr[0] = 1.0f             → CPU page fault → ioctl(0x4e) → 分配 CPU 端页面
   cudaMemset(m_ptr, ...)      → GPU page fault → ioctl(0x4e) → 迁移到 GPU 端页面

strace 对比：无预取的 UVM 分配比 ``cudaMalloc`` 多 50%+ 的 ioctl
调用，因为每次 page fault 都需要与 UVM 驱动通信。

``cudaMemPrefetchAsync`` 的作用是**提前触发批量页面迁移**，将页面从
CPU 迁移到 GPU（或反向），减少运行时的 page fault 开销。strace 显示
prefetch 后在 kernel launch 期间不再有 `0x4e` 的 page fault ioctl。

--------------

``cudaMallocAsync``：显存池与 ioctl 减免
-------------------------------------------

``cudaMallocAsync`` 是 CUDA 11.2 引入的流式分配器，从预分配的显存池
中分配内存。strace 对比：

.. code:: text

   ; cudaMalloc + cudaFree (传统)
   ioctl(0x2a) × 1   ← 分配
   ioctl(0x2b) × 1   ← fence
   ...
   ioctl(0x2a) × 1   ← 释放

   ; cudaMallocAsync + cudaFreeAsync (流式)
   ; 首次使用创建池时：
   ioctl(0x2a) × 1   ← 创建池 (初始分配 64 MB)
   ; 后续分配/释放：
   ; 无 ioctl(0x2a) — 从池中直接分配

关键：**cudaMallocAsync 通过批量预分配减少 ioctl 次数**。第一次
``cudaMallocAsync`` 触发驱动创建显存池（1 次 ioctl），后续的分配
释放都在用户态完成，不涉及系统调用。

--------------

ioctl 统计摘要
----------------

.. list-table:: 各分配路径的 ioctl 次数对比 (453 total)
   :header-rows: 1
   :widths: 25 15 15 15 40

   * - 程序段
     - ``0x2a``
     - ``0x2b``
     - ``0x4e``
     - 说明
   * - CUDA 初始化
     - 8
     - 9
     - 0
     - cuInit + cuCtxCreate
   * - ``cudaMalloc`` (8 MB)
     - 1
     - 1
     - 1
     - 分配 + fence + memset(DMA)
   * - ``cudaHostAlloc`` (8 MB)
     - 1
     - 1
     - 0
     - 注册 DMA 地址
   * - ``cudaMallocManaged`` (8 MB)
     - 1
     - 1
     - 2
     - 注册 UVM + CPU page fault + GPU page fault
   * - ``cudaHostAlloc`` Mapped
     - 1
     - 1
     - 0
     - 双映射注册
   * - ``cudaMallocAsync`` (8 MB)
     - 1
     - 1
     - 1
     - 从池分配 (额外池创建)
   * - UVM Large (64 MB)
     - 3
     - 3
     - 5
     - 多 page fault

--------------

关键发现
-----------

1. **GPU BAR 是固定的** — ``cudaMalloc`` 不产生 mmap，GPU 虚拟地址
   空间在 ``cuInit`` 阶段通过 PROT_NONE 映射一次性预留，后续分配只
   是在驱动内部建立页表。

2. **Pinned Memory = /dev/zero + DMA 注册** — 与普通 malloc 的本质区
   别是物理页面锁定和 DMA 地址注册。strace 可以清晰区分。

3. **UVM Page Fault 有显式 ioctl** — 每次 page fault 都会产生一个
   ``ioctl(0x4e)``，这是 UVM 驱动在 CPU 和 GPU 之间迁移页面的开销。
   ``cudaMemPrefetchAsync`` 用一次批量迁移替代了多次 page fault。

4. **cudaMallocAsync 减少 ioctl** — 通过显存池将多次分配合并为一次
   driver 调用。这是高频率分配/释放场景的性能关键。

5. **双映射的实现** — ``cudaHostAlloc`` Mapped 创建了一个 CPU 地址
   和一个 GPU BAR 地址映射到同一物理页面，这是通过驱动内部的页表
   别名实现的，strace 层面无额外 ioctl。

*分析基于 CUDA 13.1 / Driver 595.58.03 / NVIDIA GeForce RTX 4060 Laptop GPU。*

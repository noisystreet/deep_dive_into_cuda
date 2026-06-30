UVM Page Fault 深入分析
============================

   Unified Virtual Memory (UVM) 允许多个线程共享同一指针，系统在
   page fault 时透明迁移页面。本节通过计时、strace 和 nsys 分析
   五种 UVM 使用模式的时间开销与系统调用特征。

   :doc:`02_memory_management` 已分析 ``cudaMalloc``、pinned memory 与
   memory pool 的 ioctl 模式。UVM 在统一虚拟地址之上再叠加 **按需 page
   migration**；本节观察缺页触发的额外驱动交互。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 / RTX 4060 Laptop (512 MB 测试数据)

   测试程序: ``examples/uvm_pagefault_demo.cu`` (5 种场景)

--------------

UVM 的惰性分配模型
--------------------

``cudaMallocManaged`` 不分配物理页——它只在 UVM 驱动中注册一个虚拟地址
区间。物理页面在首次访问时按需分配。strace 可以观察到这个差异：

.. code:: text

   cudaMallocManaged(512 MB)
     → ioctl(0x2a) × 1   ← UVM 注册虚拟地址
     → 无 mmap            ← 不分配物理页
     → 无 ioctl(0x4e)     ← 不触发 DMA

   CPU touch (m[0] = 1.0f)
     → CPU page fault     ← 第一次访问触发缺页
     → ioctl(0x4e) × 1    ← 与 UVM 驱动交互
     → 分配物理页         ← 在 CPU 端分配 64 KB (GPU 页大小)

   GPU touch (cudaMemset)
     → GPU page fault     ← 第一次 GPU 访问触发缺页
     → ioctl(0x4e) × N    ← 逐页迁移或批量迁移
     → 物理页迁移到 GPU   ← 在 GPU 显存中分配新页面

strace 中的 ``ioctl(0x4e)`` 在 UVM 场景下同时服务于两种功能：普通
kernel launch (DMA) 和 page fault 处理。区分它们需要结合时间戳上下文。

--------------

各场景性能对比
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 20 25

   * - 场景
     - 耗时 (秒)
     - ioctl 次数
     - 说明
   * - CPU touch (惰性, 512 MB)
     - 0.092
     - 低
     - 物理页在 CPU 端按需分配
   * - GPU→CPU 回迁
     - 0.124
     - 中
     - 逐页迁移回 CPU
   * - 预取 GPU→CPU
     - 0.079
     - 高
     - 批量迁移，单次 ioctl
   * - 预取后 CPU 读
     - 0.003
     - 无
     - 无 page fault，纯内存读
   * - cudaMemAdvise 后 CPU 读
     - **1.974**
     - 高
     - access-by-host 需查询
   * - 交错 8×64 MB
     - 0.099
     - 中
     - 多区域交错访问

关键观察：

1. 预取使 CPU 读速度提升 40 倍（0.124 s → 0.003 s）。预取将
   页面在 GPU→CPU 回迁场景从逐页触发 fault 变为批量 DMA 迁移。

2. **cudaMemAdvise 的意外成本** — ``cudaMemAdviseSetAccessedBy``
   使 CPU 每次读都需通过驱动查询 GPU 页表（1.974 s），反而远慢于
   直接回迁（0.124 s）。这个 advise 适合**只读访问**场景——如果 CPU
   需要频繁写，手动迁移或取消 access-by 更高效。

3. **交错访问的 UVM 开销可控** — 8×64 MB 交错访问只用了 0.099 s，
   说明 UVM 驱动对多个区域的迁移请求做了合并和批量处理。

--------------

系统调用模式
--------------

不同 UVM 场景下 ioctl 的分布差异：

.. code:: text

   ; CPU touch (惰性)
   0x2a × 1   0x2b × 1   0x4e × 2    ← 初始化 + page fault

   ; GPU→CPU 回迁 (无预取)
   0x2a × 1   0x2b × 2   0x4e × 20+  ← 大量逐页 fault

   ; 预取 GPU→CPU
   0x2a × 1   0x2b × 2   0x4e × 5    ← 预取用一个 batch 替代
                                         大量逐页 fault

; cudaMemPrefetchAsync 内部
   ioctl(NV_DEV_IOCTL(0x4e), ...)     ← 批量 DMA 迁移
   ioctl(NV_DEV_IOCTL(0x2b), ...)     ← fence

--------------

UVM 性能关键点
----------------

**1. 页大小与迁移粒度**

GPU 的 UVM 页大小为 64 KB（而非 CPU 的 4 KB）。这意味着：

- 每个 page fault 覆盖 65536 字节
- stride=4096 的触页模式会浪费约 93.75% 的带宽（每 4 KB 访问触发
  64 KB 迁移）
- 顺序访问模式效率更高：每 64 KB 只触发一次 PF

**2. 预取 vs 惰性**

.. code:: text

   惰性: CPU touch → PF × N → 分配 → GPU kernel → PF × N → 迁移
   预取: CPU touch → PF × N → 分配 → Prefetch → 批量迁移 → GPU kernel → 无 PF

预取将运行时的逐页 fault 开销转移到编译期（或 launch 前）的批量迁移。
在已知访问模式时，预取可以消除几乎所有的运行时 PF 延迟。

**3. cudaMemAdvise 的正确使用**

- ``cudaMemAdviseSetPreferredLocation`` — 设置页面首选位置。如果
  数据主要被 GPU 读写，设置首选 GPU 可避免不必要的回迁。
- ``cudaMemAdviseSetAccessedBy`` — 声明设备会访问此区域。但这个
  advise 不迁移页面，只设置页表权限。如果 CPU 需要频繁读写，
  使用预取或手动迁移优于 access-by。

**4. 交错分配的线程安全成本**

多个 UVM 区域交错访问时，驱动需要维护每个区域的页表一致性。测试中
8 个 64 MB 区域交错只用了 0.099 s，说明 UVM 驱动的页表缓存和批量
处理在交错场景下表现良好——前提是访问粒度远大于页大小。

--------------

关键发现
-----------


1. **UVM 没有免 page fault 的方法** — 无论是惰性 touch 还是预取，
   物理页面必须在首次访问时分配。预取不消除 PF，只将 PF 从运行时的
   逐页触发变为启动前的批量迁移。

2. **ioctl(0x4e) 的语义过载** — 同一个 ioctl 号同时服务于 kernel
   launch、DMA 和 UVM page fault。仅凭 strace 无法区分三种用途，
   需要结合调用时机和上下文判断。

3. **cudaMemAdvise 是双刃剑** — ``SetAccessedBy`` 在不迁移页面的
   情况下允许跨设备访问，但每次访问都需要页表查询，性能可能比直接
   迁移更差。**测试中最慢的场景（1.974 s）正是使用了 access-by**。

4. **预取是最强的性能工具** — 在已知访问模式时，``cudaMemPrefetch``
   用一次批量 DMA 替代大量逐页 PF。测试中预取后 CPU 读只需 3 ms，
   是不预取回迁场景的 **1/40**。

*分析基于 CUDA 13.1 / RTX 4060 Laptop GPU / 512 MB 测试数据。实际数据
因 GPU 型号和驱动版本而异。*

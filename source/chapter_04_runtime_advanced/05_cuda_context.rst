CUDA Context 与 Driver API 深度分析
=========================================

   CUDA Context（上下文）是 Driver API 的核心抽象——它管理 GPU 的
   虚拟地址空间、模块句柄、stream 映射和显存分配。Runtime API
   （``cudaMalloc``、``<<<>>>``）隐式使用默认 context，而 Driver API
   允许显式创建和切换多个 context。本节追踪 context 的完整生命周期。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 / RTX 4060 Laptop GPU

   测试程序: ``examples/context_demo.cu``

--------------

Context 是什么？
-----------------

Context 是 GPU 的 CPU 侧 "进程"——每个 context 对应一个独立的 GPU
虚拟地址空间。等价于类比：

- CPU 进程 ↔ GPU context
- 进程的虚拟地址空间 ↔ context 的 GPU 虚拟地址空间
- 进程的打开文件表 ↔ context 的 module/cubin 列表

一个进程可以创建多个 context，每个 context 有独立的：

- GPU 虚拟地址空间（``cuMemAlloc`` 分配在这个空间内）
- 已加载的模块（``cuModuleLoad`` 的 cubin）
- Stream 和 event 映射

--------------

cuInit：Driver 初始化
-----------------------

``cuInit(0)`` 是 CUDA Driver API 的第一个调用。strace 显示它打开了
三个设备节点：

.. code:: text

   openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR)          = 23   ← 控制节点
   openat(AT_FDCWD, "/dev/nvidia-uvm", O_RDWR|O_CLOEXEC) = 24  ← UVM 节点
   openat(AT_FDCWD, "/dev/nvidia-uvm", O_RDWR|O_CLOEXEC) = 27  ← UVM 节点 (2)

关键点：

- ``/dev/nvidiactl`` 是 GPU 控制节点，提供 Device 枚举、Context 创建等
  控制功能。进程生命周期内只打开一次。
- ``/dev/nvidia-uvm`` 是 UVM 驱动节点，用于 Unified Virtual Memory
  管理。UVM 的 page fault 和迁移通过此节点通信。
- ``cuInit`` 不打开 ``/dev/nvidia0``（GPU 数据节点）——这留给
  ``cuCtxCreate``。

--------------

cuCtxCreate：上下文创建
------------------------

``cuCtxCreate`` 创建第一个 context 时，会额外打开 ``/dev/nvidia0``
（GPU 数据节点）：

.. code:: text

   cuCtxCreate(...)
     → openat("/dev/nvidia0", O_RDWR|O_CLOEXEC) = 27  ← 第一个 context
     → openat("/dev/nvidia0", O_RDWR|O_CLOEXEC) = 28  ← 2nd fd
     → openat("/dev/nvidiactl", O_RDWR|O_CLOEXEC) = 31 ← 控制 fd

     → ioctl(0x49, ...)  × N  ← context 创建 (上下文 ID 分配)
     → ioctl(0x21, ...)  × N  ← 虚拟地址空间初始化
     → ioctl(0x2a, ...)  × 8  ← 初始显存分配 (页表、内部结构)
     → ioctl(0x2b, ...)  × 8  ← fence

每个新 context 创建时，driver 会：

1. 分配一个新的 GPU 虚拟地址空间
2. 建立 PROT_NONE 映射（占位 GPU BAR 空间）
3. 创建内部的页表结构
4. 预分配一小块显存用于驱动内部管理

strace 对比：单 context vs 多 context

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - 指标
     - 单 context
     - 4 context
     - 差异
   * - ioctl 总数
     - ~400
     - 2919
     - **7×**
   * - ``0x49`` (ctx 创建)
     - 24
     - 200
     - **8×**
   * - ``0x2a`` (alloc)
     - 25
     - 863
     - **34×**
   * - ``0x2b`` (fence)
     - 25
     - 722
     - **28×**
   * - ``0x4e`` (launch)
     - 25
     - 186
     - 7×
   * - ``0x4f`` (新)
     - 0
     - 184
     - **新出现**

``0x49`` 和 ``0x4f`` 是多 context 场景下特有的 ioctl——它们分别对应
context 创建和 context 切换时的地址空间维护操作。

--------------

cuCtxPush/Pop：上下文切换
----------------------------

``cuCtxPushCurrent`` 和 ``cuCtxPopCurrent`` 切换当前线程的 active
context。strace 显示：**上下文切换不产生 ioctl**。

.. code:: text

   cuCtxPushCurrent(ctx1)    ← 无 ioctl
   cuMemAlloc(...)           ← ioctl(0x2a) 在 ctx1 的空
                               间中分配
   cuCtxPushCurrent(ctx2)    ← 无 ioctl
   cuMemAlloc(...)           ← ioctl(0x2a) 在 ctx2 的空
                               间中分配
   cuCtxPopCurrent(...)      ← 无 ioctl

上下文切换完全是用户态操作——driver 在内存中维护一个 context 栈，
``cuCtxPushCurrent`` 只是修改当前线程的 context 指针。后续所有
Driver API 调用根据这个指针选择对应的 GPU 虚拟地址空间。

交换成本就是一次函数调用的开销（微秒级），不涉及内核交互。

--------------

多 context vs 多 stream
------------------------

多 context 和多 stream 是两种不同的并发模型：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 维度
     - 多 context
     - 多 stream
   * - 虚拟地址空间
     - **独立** — 每个 context 有自
       己的地址空间
     - **共享** — 所有 stream 在同
       一地址空间内
   * - 显存分配
     - 每个 context 独立分配，不
       可互相访问
     - 所有 stream 共享显存，
       可直接访问
   * - ioctl 成本
     - **高** — 每个 context 创建
       产生 50+ ioctl
     - **无** — stream 是纯用户态
       对象
   * - 适用场景
     - 隔离性要求高（多租户、
       第三方库）
     - 同一应用的并发 kernel

--------------

关键发现
-----------

1. **Context = GPU 地址空间** — context 的核心是维护独立的 GPU 虚拟
   地址空间。每个 context 创建都触发大量的 ``ioctl(0x49)`` 和
   ``ioctl(0x2a)``，这是因为 driver 需要建立整个页表结构。

2. **上下文切换无 ioctl** — ``cuCtxPushCurrent`` 和
   ``cuCtxPopCurrent`` 纯用户态操作，不涉及内核交互。成本远低于
   context 创建。

3. **多 context 成本高昂** — 4 个 context 产生了 2919 次 ioctl
   （单 context 约 400 次），其中 ``0x49`` 和 ``0x4f`` 是多 context
   特有的 ioctl。创建 context 是**重量级操作**，不应在常规代码路径
   中频繁调用。

4. **Context 复用** — 大多数应用程序应使用 **单 context + 多 stream**
   模型而非多 context。只有当需要完全隔离的 GPU 地址空间时（如
   第三方库、多租户），才应使用多 context。

5. **三设备节点** — CUDA 使用三个不同的 ``/dev/nvidia*`` 节点：
   ``nvidiactl``（控制）、``nvidia0``（GPU 数据）、``nvidia-uvm``
   （UVM）。每种节点服务于不同目的的 ioctl。

*分析基于 CUDA 13.1 / Driver 595.58.03。多 context 场景包含 4 个独立 context。*

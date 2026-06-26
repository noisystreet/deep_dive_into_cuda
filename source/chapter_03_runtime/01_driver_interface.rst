GPU 驱动接口分析：从 CUDA API 到内核驱动
========================================

   通过 strace 跟踪 CUDA 程序 ``vector_add`` 的完整运行时， 揭示从 CUDA
   Runtime API → Driver API → 内核驱动的完整调用链

   环境: CUDA 13.1 / Driver 595.58.03 / Linux x86_64

--------------

运行时软件栈概览
-------------------

::

   用户程序 (vector_add)
         │
         ├── cudaMalloc / cudaMemcpy / cudaLaunch         ← CUDA Runtime API (libcudart)
         │         │
         │         ▼
         │    cuMemAlloc / cuMemcpyHtoD / cuLaunchKernel   ← CUDA Driver API (libcuda.so)
         │         │
         │         ▼
         │    ioctl() / mmap() / open()                    ← 系统调用接口
         │         │
         │         ▼
         │    nvidia.ko (内核模块)                          ← GPU 硬件抽象层
         │         │
         │         ▼
         │    GPU (物理硬件)
         └──────────────────────────────────────────────

三层 API 的大小对比
~~~~~~~~~~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| 层              | 文件            | 大小            | 函数数量        |
+=================+=================+=================+=================+
| **libcudart**   | ``libcu         | 4.5 MB          | ~150            |
| (Runtime API)   | dart_static.a`` |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| **libcuda**     | `               | 87 MB           | **969**         |
| (Driver API)    | `libcuda.so.1`` |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| **nvidia.ko**   | ``/lib/modules/ | **~100 MB**     | 通过 ioctl 访问 |
| (内核驱动)      | .../nvidia.ko`` |                 |                 |
+-----------------+-----------------+-----------------+-----------------+

**关键发现**: 内核模块 nvidia.ko 的大小约 100 MB，远超 libcuda.so 的 87
MB。

--------------

设备节点与初始化流程
-----------------------

三个 NVIDIA 设备节点
~~~~~~~~~~~~~~~~~~~~~~~~

::

   /dev/nvidiactl   ← 控制节点 (fd 24): GPU 控制、上下文管理、版本查询
   /dev/nvidia0     ← GPU 0 计算节点 (fd 27-50): 内存分配、kernel 执行、同步
   /dev/nvidia-uvm  ← Unified Virtual Memory (fd 25): UVM 管理、页面错误处理

初始化序列
~~~~~~~~~~~~~~

从 strace 日志中提取的完整初始化流程：

::

   ▼ Phase 1: 用户程序启动
     mmap(NULL, 2511040, ...) = 0x7f41c5200000          ← 加载 libcudart_static.a
     mmap(NULL, 91739432, ...) = 0x7f41bea00000          ← 加载 libcuda.so (87 MB)
     mmap(NULL, 2047568, ...) = 0x7f41c500c000            ← 加载 libcuda.so 的 code sections

   ▼ Phase 2: CUDA 初始化 (cuInit)
     openat(..., "/dev/nvidiactl", O_RDWR)                ← 打开控制节点
     openat(..., "/dev/nvidia-uvm", O_RDWR|O_CLOEXEC)    ← 打开 UVM 节点
     ioctl(24, NV_CTRL_IOCTL(0xd6), ...)                  ← 查询驱动版本
     ioctl(24, NV_CTRL_IOCTL(0xc8), ...)                  ← 查询 GPU 数量/信息
     ioctl(24, NV_CTRL_IOCTL(0x2b), ...)                  ← 初始化 GPU 状态
     ioctl(24, NV_CTRL_IOCTL(0x2a), ...) × N              ← 分配资源 (N=161 次)
     openat(..., "/proc/driver/nvidia/params", ...)        ← 读取驱动参数

   ▼ Phase 3: 上下文创建 (cuCtxCreate)
     openat(..., "/dev/nvidia0", O_RDWR|O_CLOEXEC)       ← 打开 GPU 0
     openat(..., "/dev/nvidiactl", O_RDWR|O_CLOEXEC)     ← 重新打开控制节点
     ioctl(28, NV_DEV_IOCTL(0xc9), ...)                   ← 创建 CUDA 上下文
     ioctl(28, NV_DEV_IOCTL(0xd7), ...)                   ← 设置上下文参数
     ioctl(28, NV_DEV_IOCTL(0x27), ...) × 4               ← 分配上下文资源
     mmap(0x200000000, 4297064448, PROT_NONE, ...)        ← 预留 4 GB GPU BAR 地址空间

   ▼ Phase 4: 内存分配 (cudaMalloc)
     ioctl(fd, NV_DEV_IOCTL(0x2a), ...)                   ← 分配 GPU 内存 (8 MB)
     mmap(NULL, 4198400, PROT_READ|PROT_WRITE, ...)       ← host 端 pinned memory (4 MB × 3)

   ▼ Phase 5: 数据拷贝 (cudaMemcpy)
     ioctl(fd, NV_DEV_IOCTL(0x4e), ...) × 25              ← DMA 传输控制

   ▼ Phase 6: Kernel Launch (cudaLaunch)
     ioctl(fd, NV_DEV_IOCTL(0x4e), ...)                   ← 提交 kernel 到 GPU 命令队列
     eventfd2(0, EFD_CLOEXEC|EFD_NONBLOCK)                 ← 创建完成通知事件
     clone3(...)                                            ← 创建 GPU worker 线程 (4 次)

   ▼ Phase 7: 同步 (cudaDeviceSynchronize)
     ioctl(fd, NV_DEV_IOCTL(0x2b), ...)                    ← 等待 GPU 完成
     futex(...)                                              ← 用户态同步

   ▼ Phase 8: 清理
     cudaFree → ioctl(NV_DEV_IOCTL(0x2b), ...)
     cuCtxDestroy → close(fd), munmap(...)

--------------

IOCTL 接口深度分析
---------------------

IOCTL 命令结构
~~~~~~~~~~~~~~~~~~

所有的 NVIDIA IOCTL 使用 ``0x46`` 作为 type 字段（ASCII 字符
``'F'``\ ），格式如下：

.. code:: c

   // 解码模板
   _IOC(dir, type='F'(0x46), nr, size)

IOCTL 命令分类统计
~~~~~~~~~~~~~~~~~~~~~~

从 410 次 IOCTL 调用中提取的命令分布：

======== ==== === ============== ======== =====================
命令     方向 NR  数据大小       出现次数 功能
======== ==== === ============== ======== =====================
``0x2a`` R+W  42  0x20 = 32 B    **161**  内存操作 / 资源管理
``0x2b`` R+W  43  0x30 = 48 B    **102**  同步 / 等待 / Barrier
``0x49`` NONE 73  0              **27**   GPU 心跳 / 轮询
``0x21`` NONE 33  0              **27**   未知空操作
``0x4e`` R+W  78  0x38 = 56 B    **25**   DMA / 数据传输
``0x1b`` NONE 27  0              **16**   低级别同步
``0xc9`` R+W  201 0x4 = 4 B      **10**   **上下文创建**
``0xce`` R+W  206 0x10 = 16 B    **8**    中断/事件管理
``0x17`` NONE 23  0              **8**    GPU 状态刷新
``0x29`` R+W  41  0x10 = 16 B    **5**    查询操作
``0x27`` R+W  39  0x38 = 56 B    **4**    **上下文资源分配**
``0xd6`` R+W  214 0x8 = 8 B      **2**    **版本查询**
``0xc8`` R+W  200 0x900 = 2304 B **2**    **GPU 信息查询**
``0xd7`` R+W  215 0x230 = 560 B  **1**    **上下文参数设置**
======== ==== === ============== ======== =====================

IOCTL 命令的具体作用（推断）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

控制通道 IOCTL (fd 24 = /dev/nvidiactl)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

=================== =============================================
命令                推断功能
=================== =============================================
``0xd6``            **NV_ESC_QUERY_VERSION**: 查询驱动版本号
``0xc8``            **NV_ESC_QUERY_DEVICES**: 查询 GPU 数量和属性
``0xd7``            **NV_ESC_REGISTER_FD**: 注册 GPU 文件描述符
``0x29``            查询 GPU 状态 / 能力
``0x2a`` / ``0x2b`` API 网关，路由到具体 GPU 操作
=================== =============================================

设备通道 IOCTL (fd 27-50 = /dev/nvidia0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======== =================================================
命令     推断功能
======== =================================================
``0xc9`` **NV_ESC_CREATE_CONTEXT**: 创建 CUDA 上下文
``0x27`` **NV_ESC_ALLOC_CONTEXT_DMA**: 分配上下文 DMA 资源
``0x4e`` **NV_ESC_EXEC_GPU_COMMAND**: 提交 GPU 命令缓冲区
``0xce`` 事件/中断管理
``0x49`` GPU 活性心跳检查
``0x21`` 空操作 / 延迟检测
======== =================================================

UVM 通道 IOCTL (fd 25 = /dev/nvidia-uvm)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

===================== ===========================
命令                  推断功能
===================== ===========================
``0x1`` (size=0x3000) UVM 模块初始化 (12 KB 参数)
``0x4b``              UVM 能力查询
``0x27``              UVM 禁用
===================== ===========================

.. mermaid:: ../_static/ioctl_pie.mmd

--------------

内存管理分析
---------------

GPU BAR 地址空间映射
~~~~~~~~~~~~~~~~~~~~~~~~

::

   mmap(0x200000000, 4297064448, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
     │
     ├── 起始地址: 0x200000000 (固定地址，非 NULL)
     ├── 大小: 4,297,064,448 字节 ≈ 4.0 GB  (GPU BAR 大小)
     ├── 权限: PROT_NONE (保留虚拟地址空间，尚未映射物理内存)
     └── 类型: MAP_PRIVATE|MAP_ANONYMOUS

这个 4GB 的 mmap **预留了整个 GPU BAR
的虚拟地址空间**\ ，实际物理内存通过后续的 ``ioctl(NV_DEV_IOCTL(0x2a))``
按需分配并映射。GPU 的 BAR (Base Address Register) 映射到 PCIe
地址空间，使得 CPU 可以通过内存映射访问 GPU 显存。

Pinned Memory 分配
~~~~~~~~~~~~~~~~~~~~~~

.. code:: c

   // Host 端 Pinned Memory (可被 GPU DMA 直接访问)
   mmap(NULL, 4198400, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
   // 3 次分配，各 4 MB，用于 host→device 和 device→host 数据传输缓冲区

这 3 次 4 MB 分配对应于 ``cudaMallocHost`` 或 ``cudaMemcpy`` 内部使用的
pinned memory 缓冲区。

GPU 显存分配
~~~~~~~~~~~~~~~~

::

   cudaMalloc(&d_a, 4MB)    → ioctl(NV_DEV_IOCTL(0x2a)) + mmap
   cudaMalloc(&d_b, 4MB)    → ioctl(NV_DEV_IOCTL(0x2a)) + mmap
   cudaMalloc(&d_c, 4MB)    → ioctl(NV_DEV_IOCTL(0x2a)) + mmap

驱动程序内部管理 GPU 显存，通过 ioctl 向内核模块请求分配，内核模块在 GPU
BAR 中划分物理页。

--------------

Kernel Launch 与执行
-----------------------

Kernel Launch 流程
~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: ../_static/kernel_launch_flow.mmd

同步机制
~~~~~~~~~~~~

::

   cudaDeviceSynchronize()
         │
         ├── 方法 1: ioctl(NV_DEV_IOCTL(0x2b))  ← 阻塞等待 GPU 完成
         │    驱动轮询 GPU 的 fence 寄存器值
         │
         ├── 方法 2: eventfd ← CPU 等待 GPU 发送完成中断
         │    GPU 执行完成后触发 MSI 中断
         │    nvidia.ko IRQ handler → eventfd → 用户态 futex 唤醒
         │
         └── 方法 3: futex ← 用户态轻量同步
              创建 4 个 clone3 线程池处理异步操作

多线程架构
~~~~~~~~~~~~~~

::

   clone3 (4 次)
     ├── Thread 1: GPU worker thread (命令提交队列)
     ├── Thread 2: GPU worker thread
     ├── Thread 3: GPU worker thread  
     └── Thread 4: GPU worker thread
              │
              └── eventfd + futex 等待 GPU 完成

CUDA 运行时在首次启动时创建了 4 个工作线程，用于管理异步操作。使用
``eventfd`` 作为 GPU 中断与用户态线程之间的通知通道。

--------------

Driver API 函数规模
----------------------

libcuda.so 导出了 **969 个函数符号**\ ，按功能分类：

============= ========== ============================
类别          数量(估计) 功能
============= ========== ============================
``cuMem*``    ~60        内存分配/管理/释放
``cuMemcpy*`` ~40        内存拷贝 (H2D/D2H/D2D)
``cuCtx*``    ~30        上下文管理
``cuLaunch*`` ~20        Kernel 启动
``cuModule*`` ~15        CUDA 模块 (fat binary) 加载
``cuDevice*`` ~15        设备属性查询
``cuArray*``  ~20        CUDA 数组 (纹理)
``cuEvent*``  ~10        事件/计时
``cuStream*`` ~15        流管理
``cuGraph*``  ~50        CUDA Graphs
``cuKernel*`` ~10        Kernel 属性
``cuFunc*``   ~10        函数属性设置
**其他**      >600       内部实现/版本化变体/未来特性
============= ========== ============================

关键 API 版本化
~~~~~~~~~~~~~~~~~~~

libcuda.so 使用版本化 API 设计：

.. code:: c

   // 旧版
   cuCtxCreate(...)           // 原始版本

   // 新版（带 _v2 后缀，扩展了参数）
   cuCtxCreate_v2(...)        // v2 接口
   cuCtxCreate_v3(...)        // v3 接口
   cuCtxCreate_v4(...)        // v4 接口（最新）

这种设计保证了向后兼容——旧程序调用 ``cuCtxCreate``\ （被映射到
v1），而新程序调用 ``cuCtxCreate_v4``\ 。

--------------

UVM (统一虚拟内存)
---------------------

::

   /dev/nvidia-uvm  ← UVM 内核模块 (nvidia_uvm.ko)

UVM 模块提供了 CPU 和 GPU 间的统一虚拟地址空间：

::

   nvidia_uvm           1945600  0    ← UVM 内核模块 (1.9 MB)

UVM 的核心机制： 1. CPU 分配 managed memory (``cudaMallocManaged``) 2.
GPU 访问时触发 **page fault** → UVM 处理 3. UVM 通过 PCIe DMA
传输缺失页面 4. 页面迁移策略由驱动自动管理

--------------

内核模块概览
---------------

NVIDIA 内核模块体系
~~~~~~~~~~~~~~~~~~~~~~~

::

   Module                  Size          Used by
   nvidia_uvm           1,945,600    0
   nvidia_drm             143,360    5
   nvidia_modeset       1,818,624    2  nvidia_drm
   nvidia             105,865,216    7  nvidia_uvm, nvidia_modeset

+-----------------------+-----------------------+-----------------------+
| 模块                  | 大小                  | 功能                  |
+=======================+=======================+=======================+
| **nvidia**            | **101 MB**            | 核心驱动：GPU         |
|                       |                       | 初                    |
|                       |                       | 始化、内存管理、ioctl |
|                       |                       | 处理、命令提交        |
+-----------------------+-----------------------+-----------------------+
| **nvidia_modeset**    | 1.7 MB                | 显示模                |
|                       |                       | 式设置、多显示器管理  |
+-----------------------+-----------------------+-----------------------+
| **nvidia_drm**        | 140 KB                | DRM (Direct Rendering |
|                       |                       | Manager) 接口         |
+-----------------------+-----------------------+-----------------------+
| **nvidia_uvm**        | 1.9 MB                | 统一虚拟内存、GPU     |
|                       |                       | page fault 处理       |
+-----------------------+-----------------------+-----------------------+

驱动参数
~~~~~~~~~~~~

::

   /proc/driver/nvidia/params:
     ResmanDebugLevel: 4294967295
     RmLogonRC: 1
     ModifyDeviceFiles: 1
     InitializeSystemMemoryAllocations: 1   ← 预分配系统内存
     UsePageAttributeTable: 4294967295       ← 使用 PAT
     EnableMSI: 1                            ← 启用 MSI 中断
     EnablePCIeGen3: 0                       ← PCIe Gen3 支持
     MemoryPoolSize: 0                       ← 内存池大小 (0=动态)
     ...

--------------

strace 统计摘要
------------------

====================== =============================================
指标                   值
====================== =============================================
总系统调用数           **1,008**
其中 IOCTL             **410** (41%)
其中 MMAP              ~30 (3%)
其中 OPENAT            ~40 (4%)
线程创建 (clone3)      4 次
eventfd (GPU→CPU 通知) 4 次
futex (用户态同步)     ~100+
打开的设备节点         /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm
====================== =============================================

--------------

完整调用链总结
------------------

::

   用户程序
     │
     ├── cudaMalloc()
     │     → libcudart: cudaMalloc()
     │       → libcuda: cuMemAlloc_v2()
     │         → 内核: open("/dev/nvidia0")
     │         → 内核: ioctl(NV_DEV_IOCTL(0x2a))  ← 分配 GPU 显存
     │         → 内核: mmap(GPU BAR)               ← 映射到用户地址空间
     │
     ├── cudaMemcpy(H2D)
     │     → libcudart: cudaMemcpy()
     │       → libcuda: cuMemcpyHtoD_v2()
     │         → 内核: ioctl(NV_DEV_IOCTL(0x4e))  ← DMA 传输
     │
     ├── cudaLaunch()
     │     → libcudart: cudaLaunch()
     │       → libcuda: cuLaunchKernel()
     │         → 构建 GPU command buffer
     │         → 内核: ioctl(NV_DEV_IOCTL(0x4e))  ← 推送命令
     │
     ├── cudaMemcpy(D2H)
     │     → ... (同 H2D)
     │
     ├── cudaDeviceSynchronize()
     │     → 内核: ioctl(NV_DEV_IOCTL(0x2b))
     │     → eventfd → futex
     │
     └── cudaFree()
           → 内核: ioctl(NV_DEV_IOCTL(0x2a))  ← 释放显存

从源代码到 GPU 执行的完整路径
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   vector_add.cu (源代码)
       ↓ nvcc 编译 (离线)
   vector_add (可执行文件, 含 fat binary)
       ↓ 运行时加载
   libcudart: cudaLaunch() → libcuda: cuLaunchKernel()
       ↓
   ioctl(NV_DEV_IOCTL(0x4e)) → nvidia.ko → GPU 命令队列
       ↓
   GPU Warp Scheduler → SM → 执行 SASS 指令 (LDG, FADD, STG)
       ↓
   LDG: GPU 通过 PCIe 回读 host pinned memory
   FADD: SM 中的 FP32 ALU 执行加法
   STG: 写回 GPU 显存
       ↓
   ioctl(fence check) → eventfd → futex → 用户态恢复

--------------

关键发现
------------

1. **三层延迟瀑布**: ``cudaMalloc()`` → Driver API → ``ioctl()`` →
   nvidia.ko，每一层都有显著的延迟开销

2. **4 GB 虚拟地址预留**: 即使是只分配 8 MB 显存的程序，也会预留 4 GB
   GPU BAR 地址空间

3. **410 次 IOCTL 中 161 次 (39%) 是资源管理**
   (``0x2a``)，是最频繁的操作

4. **nvidia.ko 101 MB**: 内核模块体积接近 Linux 内核本身，封装了整个 GPU
   硬件管理栈

5. **969 个 Driver API 函数**: libcuda.so 是 CUDA 生态中最复杂的动态库

6. **MSI 中断 + eventfd + futex**: GPU→CPU 通知链跨越了硬件中断 →
   内核事件 → 用户态同步的完整路径

7. **4 个 worker 线程**: CUDA 运行时维护线程池处理异步操作

8. **千倍体积差异**: 最简单的 CUDA 向量加法程序，运行时调用了 1,008
   次系统调用、410 次 IOCTL

.. mermaid:: ../_static/runtime_call_chain.mmd

--------------

*分析基于 CUDA 13.1 / Driver 595.58.03, GPU: sm_89 (Ada Lovelace)*

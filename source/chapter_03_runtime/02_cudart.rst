LIBCUDART 分析：CUDA Runtime API 库
===================================

   libcudart = CUDA Runtime library，是用户程序直接链接的 CUDA 库， 提供
   ``cudaMalloc``\ 、\ ``cudaMemcpy``\ 、\ ``cudaLaunch`` 等高层 API，

.. admonition:: 你知道吗？

   CUDA Runtime API (libcudart) 和 Driver API (libcuda) 的关系常被
   误解。简单说：**Runtime API 是在 Driver API 之上的一层包装**。
   你调用的 ``cudaMalloc`` 最终会调用 ``cuMemAlloc``，
   ``<<<>>>`` 会调用 ``cuLaunchKernel``。Runtime API 的存在是为了
   简化编程——它自动管理 context、module 初始化等繁琐操作。如果
   你只用一个 GPU、不关心上下文管理，使用 Runtime API 就够了；
   如果要做多 GPU、动态加载 cubin、或者控制 context 生命周期，
   就必须直接使用 Driver API。

   内部调用 libcuda (Driver API) 完成实际 GPU 操作

   分析基于 CUDA 13.1 (build 37061995)

--------------

基础信息
-----------

文件概况
~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| 文件            | 路径            | 大小            | 类型            |
+=================+=================+=================+=================+
| **静态库**      | ``libcu         | **1.4 MB**      | ar archive (1   |
|                 | dart_static.a`` |                 | 个 .o)          |
+-----------------+-----------------+-----------------+-----------------+
| **动态库**      | ``libcuda       | **740 KB**      | ELF shared      |
|                 | rt.so.13.1.80`` |                 | object          |
+-----------------+-----------------+-----------------+-----------------+
| **符号链接**    | ``libcudart.    |                 |                 |
|                 | so → libcudart. |                 |                 |
|                 | so.13 → libcuda |                 |                 |
|                 | rt.so.13.1.80`` |                 |                 |
+-----------------+-----------------+-----------------+-----------------+

静态库内部结构
~~~~~~~~~~~~~~~~~~

::

   libcudart_static.a:
     └── cudart_static.o           ← 单个目标文件 (所有代码合并到一个 .o 中)

与 libcudadevrt.a 不同（只有一个 device 端 .o），libcudart
的所有代码被链接器预先打包为一个大的目标文件。

符号统计
~~~~~~~~~~~~

=================================== ==========
指标                                值
=================================== ==========
总符号数 (T/t)                      **1,264**
公开 API (cuda\*)                   **429**
内部实现符号 (libcudart_static\_\*) **835**
公共 API 类别                       **~41** 类
=================================== ==========

--------------

API 全景
-----------

功能分类
~~~~~~~~~~~~

::

   cudaArray*       7     ─── CUDA 数组操作
   cudaChoose*      1     ─── 选择设备
   cudaCreate*      2     ─── 创建对象
   cudaCtx*         1     ─── 上下文管理
   cudaDestroy*     4     ─── 销毁对象
   cudaDev*         2     ─── 开发资源
   cudaDevice*      ~50   ─── 设备管理 (最大类别)
   cudaDriver*      2     ─── 驱动版本
   cudaEvent*       ~15   ─── 事件 (计时/同步)
   cudaExternal*    4     ─── 外部资源
   cudaFree*        7     ─── 内存释放
   cudaFunc*        8     ─── 函数属性设置
   cudaGet*         ~15   ─── 属性查询
   cudaGraph*       ~40   ─── CUDA Graphs
   cudaGraphics*    8     ─── 图形互操作
   cudaHost*        6     ─── Host 内存
   cudaImport*      1     ─── 导入
   cudaInit*        1     ─── 运行时初始化
   cudaIpc*         6     ─── 进程间通信
   cudaKernel*      2     ─── Kernel 属性
   cudaLaunch*      10    ─── Kernel 启动
   cudaLibrary*     2     ─── 库管理
   cudaMalloc*      12    ─── 内存分配
   cudaMem*         ~10   ─── 内存查询
   cudaMemcpy*      ~50   ─── 内存拷贝 (最大操作类别)
   cudaMemset*      6     ─── 内存置零
   cudaMipmapped*   5     ─── Mipmap 纹理
   cudaOccupancy*   5     ─── 占用率计算
   cudaPeek*        1     ─── 错误值查询
   cudaPointer*     2     ─── 指针属性查询
   cudaProfiler*    4     ─── Profiler 控制
   cadaRuntime*     5     ─── 运行时属性
   cudaSet*         4     ─── 设置操作
   cudaSignal*      1     ─── 信号
   cudaStream*      ~40   ─── 流管理
   cudaThread*      1     ─── 线程管理
   cudaUser*        1     ─── 用户对象
   cudaWait*        1     ─── 等待

核心 API 清单
~~~~~~~~~~~~~~~~~

.. code:: c

   // ─── 设备管理 ───
   cudaSetDevice(int device);                    // 设置当前设备
   cudaGetDevice(int* device);                   // 获取当前设备
   cudaGetDeviceCount(int* count);               // 获取设备数量
   cudaDeviceSynchronize(void);                  // 同步设备
   cudaDeviceReset(void);                        // 重置设备
   cudaDeviceGetAttribute(...);                  // 获取设备属性
   cudaChooseDevice(...);                        // 选择最佳设备

   // ─── 内存管理 ───
   cudaMalloc(void** devPtr, size_t size);       // GPU 显存分配
   cudaMallocHost(void** ptr, size_t size);      // Pinned 内存分配
   cudaMallocManaged(void** ptr, size_t size);   // Unified Memory 分配
   cudaMallocAsync(void** ptr, size_t size, cudaStream_t stream); // 异步分配
   cudaFree(void* devPtr);                       // GPU 显存释放
   cudaFreeHost(void* ptr);                      // Pinned 内存释放

   // ─── 内存拷贝 ───
   cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
   cudaMemcpyAsync(void* dst, const void* src, size_t count, 
                   cudaMemcpyKind kind, cudaStream_t stream);
   cudaMemcpy2D(...);                            // 2D 拷贝
   cudaMemcpy3D(...);                            // 3D 拷贝
   cudaMemcpyPeer(...);                          // P2P 拷贝

   // ─── Kernel 启动 ───
   cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                    void** args, size_t sharedMem, cudaStream_t stream);
   cudaLaunchCooperativeKernel(...);             // 协作式启动

   // ─── 流管理 ───
   cudaStreamCreate(cudaStream_t* stream);       // 创建流
   cudaStreamDestroy(cudaStream_t stream);       // 销毁流
   cudaStreamSynchronize(cudaStream_t stream);   // 同步流

   // ─── 事件 ───
   cudaEventCreate(cudaEvent_t* event);          // 创建事件
   cudaEventRecord(cudaEvent_t event, cudaStream_t stream);  // 记录事件
   cudaEventSynchronize(cudaEvent_t event);      // 同步事件
   cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);  // 计时

--------------

核心 API 实现机制
--------------------

实现模式：Runtime API → Driver API 包装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

libcudart 的核心模式是 **将 CUDA Runtime API 包装为对 Driver API
的调用**\ 。通过 ``__cudaGetProcAddress`` 动态获取 Driver API
函数指针后调用。

从字符串分析提取的关键调用链：

::

   Runtime API                     Driver API (libcuda)
   ────────────────────────────────────────────────────────
   cudaMalloc(...)            →   cuMemAlloc_v2(...)
   cudaFree(...)              →   cuMemFree_v2(...)
   cudaMemcpy(...)            →   cuMemcpyHtoD_v2 / cuMemcpyDtoH_v2 / cuMemcpyDtoD_v2
   cudaMemcpyAsync(...)       →   cuMemcpyHtoDAsync_v2 / ...
   cudaLaunchKernel(...)      →   cuLaunchKernel_ptsz(...)
   cudaDeviceSynchronize()    →   cuCtxSynchronize_v2()
   cudaStreamCreate(...)      →   cuStreamCreate(...)
   cudaEventCreate(...)       →   cuEventCreate(...)
   cudaSetDevice(...)         →   cuCtxCreate_v2(...) / cuCtxSetCurrent(...)
   cudaGetDevice(...)         →   cuCtxGetDevice_v2(...)
   cudaMallocHost(...)        →   cuMemAllocHost_v2(...)
   cudaMallocManaged(...)     →   cuMemAllocManaged(...)

cudaMalloc 实现流程
~~~~~~~~~~~~~~~~~~~~~~~

::

   cudaMalloc(&d_a, 4MB)
     │
     ├── 校验参数 (size > 0, devPtr != NULL)
     │
     ├── 检查当前 CUDA 上下文是否已创建
     │     └── 未创建 → 隐式调用 cuInit() + cuCtxCreate()
     │
     ├── __cudaGetProcAddress("cuMemAlloc_v2")  ← 获取 Driver API 函数
     │
     └── cuMemAlloc_v2(&d_a, 4MB)
           │
           ├── 检查空闲 GPU 显存
           ├── 在 GPU BAR 中划分物理页
           ├── 更新 GPU 页表
           └── 返回 GPU 虚拟地址

**关键要点**: - 第一次调用 ``cudaMalloc``
时会触发\ **隐式初始化**\ （创建 CUDA 上下文） - ``cudaMalloc`` 返回的是
**GPU 虚拟地址**\ ，CPU 不能直接解引用 - 实际物理内存分配由 Driver API →
ioctl 完成

cudaMemcpy 实现流程
~~~~~~~~~~~~~~~~~~~~~~~

::

   cudaMemcpy(d_a, h_a, 4MB, cudaMemcpyHostToDevice)
     │
     ├── 检查参数有效性
     ├── 推断方向 (HostToDevice / DeviceToHost / DeviceToDevice)
     │
     ├── 同步当前流（默认流 = 同步操作）
     │
     └── 调用相应 Driver API:
           cudaMemcpyHostToDevice   → cuMemcpyHtoD_v2(d_a, h_a, 4MB)
           cudaMemcpyDeviceToHost   → cuMemcpyDtoH_v2(h_a, d_a, 4MB)
           cudaMemcpyDeviceToDevice → cuMemcpyDtoD_v2(d_a, d_b, 4MB)
                 │
                 └── 内核: ioctl(NV_DEV_IOCTL(0x4e), DMA_descriptor)
                       └── GPU DMA 引擎通过 PCIe 传输数据

**DMA 传输路径**:

::

   HostToDevice:
     CPU RAM → PCIe (DMA) → GPU VRAM

   DeviceToHost:
     GPU VRAM → PCIe (DMA) → CPU RAM (pinned buffer → user buffer)

   DeviceToDevice:
     GPU VRAM → GPU VRAM (GPU 内部, 不经过 PCIe)

cudaLaunchKernel 实现流程
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   cudaLaunchKernel(vector_add, grid, block, args, sharedMem, stream)
     │
     ├── 1. 将 kernel 函数指针解析为 cubin 中的入口
     │     └── __cudaRegisterFunction 时建立的映射表
     │
     ├── 2. 构建 kernel 启动参数
     │     ├── gridDim.x/y/z
     │     ├── blockDim.x/y/z
     │     ├── kernel 参数 (通过 __cudaSetupArgSimple)
     │     └── 共享内存大小
     │
     ├── 3. 调用 Driver API
     │     └── cuLaunchKernel_ptsz(func, gridX, gridY, gridZ,
     │                              blockX, blockY, blockZ,
     │                              sharedMem, stream, kernelParams, extra)
     │
     └── 4. Driver 构造 GPU 命令缓冲区
           └── ioctl(NV_DEV_IOCTL(0x4e), cmd_buf)
                 └── GPU 调度器调度线程块到 SM

\__cudaRegister\* 机制
~~~~~~~~~~~~~~~~~~~~~~~~~~

所有 ``__cudaRegister*`` 函数是实现\ **编译期注册**\ 的核心：

**关键要点**: 这些函数在 **main() 之前**\ （constructor
阶段）被调用，确保程序执行时所有 kernel 已加载完毕。

Kernel Launch 内部 (从 Runtime 到 Driver)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cudaLaunchKernel`` 的进一步分解：

::

   用户代码:   kernel<<<grid, block>>>(args);
                   ↓
   cudafe++ 翻译:  kernel(args);  // 展开为 wrapper 调用
                   ↓
   wrapper:        __device_stub__Z10vector_addPKfS0_Pfi(args)
                   ↓
                   __cudaLaunchPrologue(4);        // 设置参数计数
                   __cudaSetupArgSimple(a, 0UL);   // 排列参数
                   __cudaSetupArgSimple(b, 8UL);
                   __cudaSetupArgSimple(c, 16UL);
                   __cudaSetupArgSimple(n, 24UL);
                   __cudaLaunch((char*)vector_add); // 内部调用
                   ↓
   libcudart:      cudaLaunchKernel(func, grid, block, args, 0, 0);
                   ↓
                   cuLaunchKernel_ptsz(handle, grid, block, args, ...);
                   ↓
   libcuda:        ioctl(0x4e, cmd_buffer);  // 提交到 GPU

--------------

GPU 内核模块与运行时关系
---------------------------

::

   libcudart (Runtime API)
       │
       ├── 静态链接: libcudart_static.a (1.4 MB)
       │     └── 嵌入到可执行文件中
       │
       ├── 动态链接: libcuda.so.1 (87 MB, Driver API)
       │     └── 通过 dlopen/dlsym 或直接链接
       │
       └── 内部依赖: libcudadevrt.a (1 MB, Device 端运行时)
             └── 通过 nvlink -lcudadevrt 链接到 device 代码

链接方式对比
~~~~~~~~~~~~~~~~

+---------------------------+---------------------------------+------------------+
| 方式                      | 优点                            | 缺点             |
+===========================+=================================+==================+
| **静态链接 libcudart**    | 独立可执行文件，不依赖 .so 版本 | 体积稍大         |
+---------------------------+---------------------------------+------------------+
| **动态链接 libcudart.so** | 可升级 .so 而不重编译           | 运行时需找到 .so |
+---------------------------+---------------------------------+------------------+

在我们的编译过程中，\ ``g++`` 链接时使用了 ``-lcudart_static``\ ，因此
``libcudart_static.a`` 被嵌入到最终可执行文件中。

大小对比
~~~~~~~~~~~~

.. list-table:: 大小对比
   :header-rows: 1
   :widths: 25 15 20 40

   * - 库
     - 大小
     - 位置
     - 角色
   * - **libcudart_static.a**
     - **1.4 MB**
     - CUDA lib64
     - 用户 API (Runtime)
   * - **libcuda.so.1**
     - **87 MB**
     - 系统 lib
     - 硬件接口 (Driver)
   * - **libcudadevrt.a**
     - **1.0 MB**
     - CUDA lib64
     - Device 端运行时
   * - **libnvvm.so.4**
     - **61 MB**
     - CUDA nvvm
     - 编译引擎

--------------

内部符号分析
---------------

哈希化符号保护
~~~~~~~~~~~~~~~~~~

libcudart 内部的 **835 个非公开符号** 使用哈希化名称隐藏实现细节：

::

   libcudart_static_0056bd523cc9f704dca4e91ee8c0547f1309e92f
   libcudart_static_009a103bf0390f205162ce6b644a1fc9070c5ae7
   ...

这些是静态链接时内部使用的辅助函数。SHA1 哈希化用于防止符号冲突。

\__cudaGetProcAddress 机制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c

   __cudaGetProcAddress(const char* symbolName);

这是 Runtime API 连接到 Driver API 的关键桥梁，支持按名称查找 Driver API
函数指针。这种方式使得 Runtime API 可以适配不同版本的 Driver API。

\_ptsz 后缀
~~~~~~~~~~~~~~~

Runtime API 中存在大量 ``_ptsz`` 后缀变体：

.. code:: text

   cudaLaunchKernel_ptsz(...)    ← "per-thread default stream" 版本
   cudaMemcpyAsync_ptsz(...)
   cudaStreamSynchronize_ptsz(...)

``_ptsz`` = Per-Thread Stream per Default。在 CUDA 7+ 中，每个 host
线程拥有独立的默认流，\ ``_ptsz`` 变体实现了这个特性。对应的 ``_ptds``
变体则是旧式的 per-process default stream。

--------------

运行时状态管理
-----------------

libcudart 内部维护了运行时状态的全局变量：

::

   内部状态 (推测):
     ┌─ 当前 CUDA 上下文 (每个线程)
     ├─ 已注册的 fat binary 列表
     ├─ 已注册的 kernel 函数表
     ├─ 已注册的 device 变量表
     ├─ 默认流 (per-thread)
     ├─ 当前设备 ID
     ├─ 设备属性缓存
     ├─ 错误码 (cudaGetLastError)
     └─ 内存池状态 (cudaMallocAsync)

--------------

与其他组件的关系总览
-----------------------

::

   编译时:                                     运行时:
   ─────────                                  ─────────
   vector_add.cu                              vector_add (可执行文件)
        │                                          │
        ▼                                          ▼
   nvcc (驱动编译器)                           libcudart (1.4 MB, 静态嵌入)
        │                                          │
        ├── cudafe++ (CUDA 前端)                    ├── cudaMalloc(...)
        │     └── 生成 stub (含 __cudaRegister*)    │     └── cuMemAlloc_v2(...)
        │                                          │           └── ioctl(0x2a)
        ├── cicc (CUDA→PTX)                        │
        │     └── 生成 .ptx                         ├── cudaMemcpy(...)
        │                                          │     └── cuMemcpyHtoD(...)
        ├── ptxas (PTX→SASS)                       │           └── ioctl(0x4e)
        │     └── 生成 .cubin                       │
        │                                          ├── cudaLaunchKernel(...)
        ├── fatbinary (打包)                        │     └── cuLaunchKernel(...)
        │     └── 生成 .fatbin.c                    │           └── ioctl(0x4e)
        │                                          │
        ├── g++ (链接)                             └── cudaDeviceSynchronize()
        │     └── -lcudart_static                       └── cuCtxSynchronize_v2()
        │           └── 嵌入到 vector_add                      └── ioctl(0x2b)
        │
        └── nvlink (device 链接)
              └── -lcudadevrt

--------------

关键发现
-----------

1. **薄包装层**: libcudart (1.4 MB) 是对 libcuda (87 MB)
   的\ **薄包装**\ 。大部分 ``cuda*`` 函数就是调用对应的 ``cu*`` Driver
   API。

2. **静态单对象**: ``libcudart_static.a`` 只包含 1 个 ``.o`` 文件
   (cudart_static.o)，所有代码预先链接为一个单元。

3. **429 个公开 API**: 涵盖设备管理、内存分配、内存拷贝、Kernel
   启动、流、事件、Graph 等完整功能。

4. **哈希化内部符号**: 835 个内部实现符号全部使用了 SHA1 哈希化命名
   (``libcudart_static_*``)。

5. **隐式初始化**: 首次调用 ``cudaMalloc`` 时自动触发 ``cuInit`` +
   ``cuCtxCreate``\ 。

6. \**_ptsz 后缀*\*: CUDA 7+ 的特性，每个 host 线程拥有独立的默认流。

7. \**\__cudaGetProcAddress*\*: 动态获取 Driver API
   函数指针的桥梁，支持跨版本兼容。

8. **835/1264 (66%) 是内部符号**: 公开 API 只占约 1/3，而内部实现占了
   2/3。

--------------

*分析基于 CUDA 13.1 (build 37061995)*

Kernel Launch 深度追踪
========================

   聚焦 ``vector_add<<<blocks, threads>>>(...)`` 这一行代码，从反汇编、
   stub 源码与 strace 三条线索，逐步还原 kernel 启动的完整调用链。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) / Linux x86-64

   示例: ``examples/vector_add.cu`` 第 40 行

--------------

与前两节的关系
----------------

:doc:`01_driver_interface` 从全局 strace 视角列出了 kernel launch 阶段的
``ioctl(0x4e)``；:doc:`02_cudart` 概括了 ``cudaLaunchKernel`` 的内部流程。
本节在此基础上 **逐步追踪**——用 ``objdump``\ 、\ ``nm`` 和 stub 中间文件
对齐每一层函数名，回答三个具体问题：

1. ``<<<4096, 256>>>`` 的配置信息存在哪里、何时被读取？
2. ``vector_add`` 函数指针如何关联到 fat binary 中的 SASS 入口？
3. ``cudaLaunchKernel`` 最终通过哪条 Driver API 触发 ``ioctl``？

分析工具
--------

=================== ========================================
工具                用途
=================== ========================================
``objdump -d``      反汇编 host 侧 launch 路径
``nm -C``           列出 stub / 注册 / launch 相关符号
``readelf -S/-x``   检查 ``.nv_fatbin`` 段与 constructor 表
``strace -e ioctl`` 观察 launch 对应的 ``0x4e`` 命令提交
stub 中间文件       ``vector_add.cudafe1.stub.c``（``--keep`` 编译）
=================== ========================================

复现命令：

.. code:: bash

   cd examples && mkdir -p build && cd build
   cmake .. && make
   nm -C vector_add | rg 'Launch|Register|device_stub|vector_add'
   objdump -d vector_add | rg 'PushCall|PopCall|device_stub|LaunchKernel'
   strace -f -e trace=ioctl ./vector_add 2>&1 | rg '0x4e'

--------------

Launch 配置：<<<>>> 去了哪里？
--------------------------------

源码中的 launch 语句：

.. code:: cuda

   int threads = 256;
   int blocks = (n + threads - 1) / threads;   // n=1<<20 → blocks=4096
   vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

``n = 1\,048\,576`` 时，grid 为 **4096 × 1 × 1**，block 为 **256 × 1 × 1**。
NVCC 不会把 ``<<<>>>`` 保留到运行时——它拆成两步：

1. **Push 配置** — ``main`` 在调用 kernel 前执行
   ``__cudaPushCallConfiguration(grid, block, sharedMem, stream)``
2. **Pop 配置** — stub 内部调用 ``__cudaPopCallConfiguration`` 取回配置

``main`` 中的实际指令序列（``objdump`` 摘录）：

.. code:: text

   8e1b:  call  __cudaPushCallConfiguration    ; push grid/block
   8e36:  call  _Z10vector_addPKfS0_Pfi        ; 调用 wrapper → stub
   8e3b:  call  cudaDeviceSynchronize         ; 等待 GPU 完成

Push/Pop 机制把 **launch 配置** 与 **kernel 参数** 解耦：配置走隐式栈，四个
指针/int 参数走 stub 内的数组。

--------------

编译期注册：main 之前发生了什么？
------------------------------------

在 ``main()`` 执行任何 CUDA API 之前，链接进可执行文件的 constructor 已完成
fat binary 注册。

stub 中间文件 ``vector_add.cudafe1.stub.c`` 中的注册逻辑：

.. code:: c

   static void __nv_cudaEntityRegisterCallback(void **__T7) {
       __nv_dummy_param_ref(__T7);
       __nv_save_fatbinhandle_for_managed_rt(__T7);
       __cudaRegisterEntry(__T7,
           (void (*)(const float*, const float*, float*, int))vector_add,
           _Z10vector_addPKfS0_Pfi, (-1));
   }
   static void __sti____cudaRegisterAll(void) {
       __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);
   }

``readelf -x .init_array`` 显示 constructor 表包含 ``0x9329``\ ，对应符号
``__sti____cudaRegisterAll``。注册链：

.. code:: text

   __sti____cudaRegisterAll()
     └── __cudaRegisterBinary(callback)
           └── __cudaRegisterFatBinary()      ← 加载 .nv_fatbin 段
           └── __cudaRegisterEntry()          ← 关联 host 函数 ↔ device 符号
                 └── __cudaRegisterFunction() ← 解析 mangled name → CUfunction

注册完成后，Runtime 内部维护一张映射表：host 侧的 ``vector_add`` 函数指针
→ fat binary 中 ``_Z10vector_addPKfS0_Pfi`` 的 SASS 入口。Launch 时无需再解
析 ELF——``__cudaGetKernel`` 直接查表。

相关符号（``nm -C vector_add`` 摘录）：

.. code:: text

   __cudaRegisterFatBinary
   __cudaRegisterFunction
   __cudaRegisterEntry (via callback)
   __sti____cudaRegisterAll

Fat binary 段信息：

.. code:: text

   [18] .nv_fatbin        PROGBITS   000000000009a230
   [31] .nvFatBinSegment  PROGBITS   00000000000b5558

--------------

Device Stub：<<<>>> 的真正执行体
----------------------------------

cudafe++ 为每个 ``__global__`` kernel 生成 stub。源码级 stub（单行展开）：

.. code:: c

   void __device_stub__Z10vector_addPKfS0_Pfi(
       const float *__par0, const float *__par1, float *__par2, int __par3)
   {
       __cudaLaunchPrologue(4);
       __cudaSetupArgSimple(__par0, 0UL);
       __cudaSetupArgSimple(__par1, 8UL);
       __cudaSetupArgSimple(__par2, 16UL);
       __cudaSetupArgSimple(__par3, 24UL);
       __cudaLaunch((char *)vector_add);
   }

``__cudaLaunchPrologue`` / ``__cudaSetupArgSimple`` 是 ``host_runtime.h`` 中的
宏，在 CUDA 13 中展开为更底层的参数打包逻辑。反汇编揭示了 stub 的实际步骤：

**步骤 1 — 构建 kernel 参数指针数组**

.. code:: text

   90e8–9129:  将 &d_a, &d_b, &d_c, &n 的地址依次写入栈上数组 kernelParams[]

参数布局遵循 Itanium ABI 偏移：三个 ``float*`` 各占 8 字节，``int n`` 在偏移
24 处。

**步骤 2 — 懒加载 CUkernel handle**

.. code:: text

   9173–917d:  lea  __handle; lea vector_add; call __cudaGetKernel
               ; 首次调用时解析 handle，后续走 guard 快速路径

``__cudaGetKernel`` 查注册表，将 host 函数指针 ``vector_add`` 转换为
``CUkernel_st*`` handle，缓存在 ``__handle`` 静态变量中。

**步骤 3 — 弹出 launch 配置并发起 launch**

.. code:: text

   91d9:  call __cudaPopCallConfiguration    ; 取 grid/block/stream
   922a:  call __cudaLaunchKernel_helper      ; CUkernel + dim3 + params

``__cudaLaunchKernel_helper`` 的函数签名（反汇编符号名）：

.. code:: c

   __cudaLaunchKernel_helper(
       CUkernel_st *kernel,     // rdi
       dim3 gridDim,            // rsi, rdx, ...
       dim3 blockDim,
       void **kernelParams,
       size_t sharedMem,
       CUstream_st *stream);

--------------

Runtime 层：__cudaLaunchKernel → cudaLaunchKernel
--------------------------------------------------

``__cudaLaunchKernel_helper`` 末尾调用 ``__cudaLaunchKernel``\ ，后者仅 10 字
节——是一条 **tail jump**：

.. code:: text

   __cudaLaunchKernel:
     1dfb5:  jmp  cudaLaunchKernel        ; 直接跳转，无额外逻辑

``cudaLaunchKernel`` 本身约 740 字节（``nm`` 显示），负责：

1. 获取当前 CUDA 上下文（内部哈希化辅助函数）
2. 校验参数合法性
3. 通过 ``__cudaGetProcAddress`` 动态解析 ``cuLaunchKernel_ptsz`` 函数指针
4. 调用 Driver API

libcuda.so 导出符号（``nm -D libcuda.so.1``）：

.. code:: text

   cuLaunchKernel
   cuLaunchKernel_ptsz        ← Runtime 实际调用的版本
   cuLaunchKernelEx
   cuLaunchKernelEx_ptsz

``_ptsz`` 后缀表示 **Per-Thread default Stream**——CUDA 7+ 每个 host 线程拥
有独立默认流（详见 :doc:`02_cudart` 中的 ``_ptsz`` 说明）。

--------------

Driver 层：ioctl 命令提交
--------------------------

:doc:`01_driver_interface` 已分析 ``ioctl`` 命令格式。Kernel launch 最终通过
``NV_ESC_EXEC_GPU_COMMAND``（NR = ``0x4e``）提交：

.. code:: text

   ioctl(fd, _IOC(R|W, 0x46, 0x4e, 0x38), argp)
   // type='F', nr=0x4e, size=56 字节

对 ``vector_add`` 完整运行（含 malloc/memcpy/launch/sync/free）的统计：

================== ======
指标               值
================== ======
``ioctl(0x4e)``    **25** 次
``ioctl(0x2a)``    161 次（内存管理）
``ioctl(0x2b)``    102 次（同步）
设备 fd            ``/dev/nvidia0`` (fd=10)
================== ======

``0x4e`` 是 **通用 GPU 命令通道**——既承载 ``cudaMemcpy`` 的 DMA 描述符，也
承载 kernel dispatch 命令。一次 launch 通常在该通道上产生 **连续多次**
``ioctl(0x4e)``\ ：Driver 分批提交 grid 配置、参数地址、SASS 入口 PC 等子命
令。strace 中可观察到同一 ``argp`` 地址被重复提交（fence 轮询模式）。

Launch 之后，Runtime 创建 ``eventfd`` 并 ``clone3`` 启动 worker 线程（:doc:
`01_driver_interface` 中 Phase 6–7），用于异步完成通知；``cudaDeviceSynchronize``
则通过 ``ioctl(0x2b)`` 阻塞等待 GPU fence。

.. mermaid:: ../_static/kernel_launch_flow.mmd

--------------

完整调用链
----------

.. mermaid:: ../_static/kernel_launch_call_stack.mmd

按时间顺序串起来：

.. code:: text

   [程序启动]
     __sti____cudaRegisterAll()          ← constructor
       → __cudaRegisterBinary()
         → 加载 .nv_fatbin → 注册 _Z10vector_addPKfS0_Pfi

   [main 中]
     __cudaPushCallConfiguration(4096,1,1, 256,1,1, 0, 0)
     vector_add(d_a, d_b, d_c, n)
       → __device_stub__Z10vector_addPKfS0_Pfi(...)
           → 构建 kernelParams[4]
           → __cudaGetKernel(&handle, vector_add)     ← 首次查表
           → __cudaPopCallConfiguration(&grid, &block, &stream)
           → __cudaLaunchKernel_helper(handle, ...)
               → __cudaLaunchKernel()
                 → cudaLaunchKernel()
                   → cuLaunchKernel_ptsz(...)          ← libcuda.so
                     → ioctl(0x4e, cmd_buf)           ← nvidia.ko
                       → GPU 命令队列
                         → 4096 个 block × 256 线程
                           → SM 执行 SASS (_Z10vector_addPKfS0_Pfi)

     cudaDeviceSynchronize()
       → cuCtxSynchronize_v2()
         → ioctl(0x2b) + eventfd/futex

--------------

与编译产物的对应
----------------

Launch 链条的每一环都对应编译阶段的某一产物：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Launch 阶段
     - 编译产物
   * - ``__cudaRegisterBinary``
     - ``.nv_fatbin`` 段、``vector_add.fatbin.c``
   * - ``__cudaGetKernel``
     - mangled 名 ``_Z10vector_addPKfS0_Pfi``
   * - ``cuLaunchKernel`` 参数
     - stub 中 ``kernelParams`` 布局
   * - GPU 执行体
     - ``vector_add.sm_89.cubin`` 中 22 条 SASS

PTX 入口（:doc:`../chapter_01_compilation/02_intermediate_files`）：

.. code:: text

   .visible .entry _Z10vector_addPKfS0_Pfi(
       .param .u64 ..._param_0,    // a
       .param .u64 ..._param_1,    // b
       .param .u64 ..._param_2,    // c
       .param .u32 ..._param_3     // n
   )

Driver 提交 launch 命令时，参数类型/偏移必须与 PTX ``.param`` 声明一致——
stub 中 ``__cudaSetupArgSimple`` 的偏移 (0, 8, 16, 24) 正是为此对齐。

--------------

关键发现
--------

1. ``<<<>>>`` 被拆为 Push + Pop — 配置不经过函数参数，而由
   ``__cudaPushCallConfiguration`` / ``__cudaPopCallConfiguration`` 传递。

2. stub 是 launch 的真正入口 — 用户调用的 ``vector_add(...)`` 只是
   wrapper；``__device_stub__...`` 负责参数打包与 launch 发起。

3. 注册发生在 main 之前 — ``__sti____cudaRegisterAll`` constructor 将
   fat binary 加载并建立 host→device 符号映射，launch 时 ``__cudaGetKernel``
   只需 O(1) 查表。

4. 三层函数跳转 — ``__cudaLaunchKernel_helper`` → ``__cudaLaunchKernel``
   (tail jmp) → ``cudaLaunchKernel`` → ``cuLaunchKernel_ptsz``\ ，Runtime 层
   本质是参数校验 + 动态符号解析。

5. ``ioctl(0x4e)`` 多用途 — 同一命令号服务 DMA 拷贝与 kernel dispatch；不能
   仅凭 NR 区分操作类型，需结合调用时机与上下文。

6. 4096 blocks 一次提交 — grid 配置 ``(4096, 1, 1)`` 作为整体写入命令
   缓冲区，由 GPU Warp Scheduler 逐 block 调度到 SM。

--------------

*Deep Dive Into CUDA — 2026 年 6 月*

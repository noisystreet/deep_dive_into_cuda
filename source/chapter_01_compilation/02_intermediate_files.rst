NVCC 编译中间产物分析
=====================

   通过 ``nvcc --keep`` 保留编译过程的所有中间文件，逐一分析其角色和内容

   环境: CUDA 13.1 / sm_89 (Ada Lovelace)

   :doc:`01_compilation_pipeline` 从 ``nvcc --verbose`` 概括了 11 步编译链。
   本节启用 ``--keep``，把日志中的抽象步骤落实为磁盘上的 ``.ptx``、
   ``.cubin``、``.fatbin.c`` 等可分析文件。

.. admonition:: 你知道吗？

   当 CUDA kernel 出现"未定义行为"级别的错误（如意外写坏内存、PTX
   JIT 失败），最有效的定位方式是 ``nvcc --keep``。保留的 ``.ptx``
   文件可以单独用 ``ptxas --info`` 检查，``.cubin`` 可以用
   ``cuobjdump -sass`` 反汇编查看实际硬件指令。CUDA 编译器不会像
   GCC/Clang 那样输出详细优化报告，中间文件是唯一的"编译器日志"。

.. admonition:: 你知道吗？

   虽然日常开发不会每次都用 ``--keep``（因为它大幅增加编译时间），
   但以下场景中它是标准做法：(1) 提交 CUDA bug report 给 NVIDIA 时，
   ``--keep`` 的输出是必附内容；(2) 分析性能问题时，查看 ptxas 生成
   的 SASS 指令数可以帮助确定寄存器溢出；(3) 调试 ``cudaMemset``
   未按预期工作的底层问题时，fatbin 的 ELF 段布局是关键线索。

中间产物总览
------------

.. list-table:: 中间产物总览
   :header-rows: 1
   :widths: 35 15 20 30

   * - 文件
     - 大小
     - 生成阶段
     - 角色
   * - ``vector_add.cpp1.ii``
     - 1.3 MB
     - device 预处理
     - 设备端预处理的 C++ 源文件
   * - ``vector_add.cpp4.ii``
     - 1.2 MB
     - host 预处理
     - 主机端预处理的 C++ 源文件
   * - ``vector_add.cudafe1.gpu``
     - 15 KB
     - cudafe++
     - 提取出的 device 端代码（含 CUDA 内部头文件展开）
   * - ``vector_add.cudafe1.cpp``
     - 1.1 MB
     - cudafe++
     - 翻译后的 host 端 C++ 代码
   * - ``vector_add.cudafe1.stub.c``
     - 1.7 KB
     - cudafe++
     - device 函数的 host stub 和注册代码
   * - ``vector_add.cudafe1.c``
     - 65 B
     - cudafe++
     - 辅助 C 文件
   * - ``vector_add.ptx``
     - 1.3 KB
     - cicc
     - PTX (Parallel Thread Execution) 虚拟汇编
   * - ``vector_add.sm_89.cubin``
     - 3.5 KB
     - ptxas
     - GPU 二进制机器码 (SASS)
   * - ``vector_add.fatbin.c``
     - 11 KB
     - fatbinary
     - fat binary 嵌入为 C 数组
   * - ``vector_add.fatbin``
     - 4 KB
     - fatbinary
     - fat binary 原始文件
   * - ``vector_add.module_id``
     - 35 B
     - cudafe++
     - 模块 ID



预处理阶段: ``.cpp1.ii`` vs ``.cpp4.ii``
----------------------------------------------
nvcc 会进行两次预处理，分别用于 host 和 device 编译路径：

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - 方面
     - ``vector_add.cpp4.ii`` (host)
     - ``vector_add.cpp1.ii`` (device)
   * - 预定义宏
     - ``-D__CUDA_ARCH_LIST__=890``
     - ``-D__CUDA_ARCH__=890 -D__CUDA_ARCH_LIST__=890 -DCUDA_DOUBLE_MATH_FUNCTIONS``
   * - 目标
     - 用于 ``cudafe++`` 解析 CUDA 语法
     - 用于 ``cicc`` 编译为 PTX
   * - 内容区别
     - 不包含 ``__CUDA_ARCH__`` 宏
     - 包含 ``__CUDA_ARCH__=890``，会展开架构相关的条件代码

关键区别在于 ``__CUDA_ARCH__`` 宏——它在 device 路径中定义（值为 890 即
sm_89），在 host 路径中未定义。这允许同一个 ``.cu`` 文件中使用
``#ifdef __CUDA_ARCH__`` 编写主机/设备共享的代码。

源码到预处理的流变示例
^^^^^^^^^^^^^^^^^^^^^^

原始 CUDA 源码:

.. code:: cuda

   __global__ void vector_add(const float *a, const float *b, float *c, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           c[idx] = a[idx] + b[idx];
       }
   }

在预处理文件中，\ ``__global__`` 等关键字被展开为编译器内部标识，
``blockIdx``\ 、\ ``threadIdx`` 等内置变量变为对内部数据结构的引用。
同时 ``<cuda_runtime.h>`` 等大量头文件被展开，使文件膨胀至 1.2MB+。



cudafe++ 阶段: 核心语法分离
---------------------------

``cudafe++`` 是 CUDA 编译最核心的工具。它读取预处理后的
``.cpp4.ii``，输出三组文件：

``vector_add.cudafe1.gpu`` — 提取的 device 代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
这个 15KB 的文件包含了 **所有需要在 GPU 上运行的代码**，包括：

- **我们的 kernel 函数**\ （已展开为 C 风格，保留了行号信息）
- **CUDA 内部 device 函数**\ （``cudaMalloc`` 在 device 端的 stub、
  ``dim3`` 构造函数等）
- **头文件展开**\ （``device_runtime.h``\ 、``device_functions.h``\ 、
  ``common_functions.h`` 等）

关键内容示例（原始源码部分）：

.. code:: c

   __global__ __var_used__ void _Z10vector_addPKfS0_Pfi(
       const float *a, const float *b, float *c, int n)
   {
       int __cuda_local_var_37641_9_non_const_idx;
       __cuda_local_var_37641_9_non_const_idx =
           ((int)(((blockIdx.x) * (blockDim.x)) + (threadIdx.x)));
       if (__cuda_local_var_37641_9_non_const_idx < n)
       {
           (c[__cuda_local_var_37641_9_non_const_idx]) =
               ((a[__cuda_local_var_37641_9_non_const_idx]) +
                (b[__cuda_local_var_37641_9_non_const_idx]));
       }
   }

注意：

- 函数名被 mangled 为 ``_Z10vector_addPKfS0_Pfi``\ （符合 Itanium C++
  ABI）
- 局部变量被重命名为 ``__cuda_local_var_...`` 形式
- ``blockIdx.x`` 等被保留，由后续 cicc 编译器处理

``vector_add.cudafe1.stub.c`` — Host stub 和注册
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
这个 1.7KB 的文件是 **cudafe++ 最重要的输出** 之一。它包含了：

.. code:: c

   // kernel 的 host stub：将 kernel 调用翻译为 CUDA 运行时 API
   __attribute__((visibility("hidden"))) void __device_stub__Z10vector_addPKfS0_Pfi(
       const float *__par0, const float *__par1, float *__par2, int __par3)
   {
       __cudaLaunchPrologue(4);           // 设置 4 个参数
       __cudaSetupArgSimple(__par0, 0UL);  // 参数 0
       __cudaSetupArgSimple(__par1, 8UL);  // 参数 1
       __cudaSetupArgSimple(__par2, 16UL); // 参数 2
       __cudaSetupArgSimple(__par3, 24UL); // 参数 3
       __cudaLaunch((char *)vector_add);   // 启动 kernel
   }

   // wrapper 函数，用户调用的 vector_add 实际指向这里
   void vector_add(const float *__cuda_0, const float *__cuda_1,
                   float *__cuda_2, int __cuda_3)
   {
       __device_stub__Z10vector_addPKfS0_Pfi(__cuda_0, __cuda_1, __cuda_2, __cuda_3);
   }

   // 注册回调：程序启动时自动调用，将 kernel 注册到 CUDA 驱动
   static void __nv_cudaEntityRegisterCallback(void **__T7) {
       __nv_dummy_param_ref(__T7);
       __nv_save_fatbinhandle_for_managed_rt(__T7);
       __cudaRegisterEntry(__T7,
           ((void (*)(const float *, const float *, float *, int))vector_add),
           _Z10vector_addPKfS0_Pfi, (-1));
   }

   // 构造函数：在 main() 之前执行，完成 CUDA 初始化
   static void __sti____cudaRegisterAll(void) {
       __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);
   }

**要点**：

- ``vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n)`` 被展开为对
  ``__device_stub__...`` 的调用
- stub 内部使用 ``__cudaLaunchPrologue`` / ``__cudaSetupArgSimple`` /
  ``__cudaLaunch`` 这套底层 API
- ``__sti____cudaRegisterAll`` 是一个
  ``__attribute__((__constructor__))``\ ，在 ``main()`` 之前自动调用，完成
  kernel 注册
- ``__cudaRegisterEntry`` 将 kernel 名称（mangled 的
  ``_Z10vector_addPKfS0_Pfi``）与 fat binary 中的 device 代码关联

``vector_add.cudafe1.cpp`` — Host C++ 代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
这个 1.1MB 的文件是 ``cudafe++`` 输出的 **纯 C++ 版本** 的 host 代码。它
不再包含任何 CUDA 语法（\ ``<<<>>>``\ 、\ ``__global__`` 等已被翻译），
可以用标准 C++ 编译器（gcc）编译。

主要内容包括：

- CUDA 头文件的纯 C++ 展开（\ ``cuda_runtime.h``\ 、\ ``driver_types.h`` 等）
- stub 函数（与 ``.stub.c`` 相同）
- 所有 host 端代码



Device 编译工具链
-----------------

``vector_add.ptx`` — PTX 虚拟汇编
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cicc 将 device 代码编译为 PTX，这是 GPU 指令集的虚拟中间表示，共 55 行：

.. code:: text

   .version 9.1
   .target sm_89
   .address_size 64

   .visible .entry _Z10vector_addPKfS0_Pfi(
       .param .u64 _Z10vector_addPKfS0_Pfi_param_0,   // a (指针, 64位)
       .param .u64 _Z10vector_addPKfS0_Pfi_param_1,   // b (指针, 64位)
       .param .u64 _Z10vector_addPKfS0_Pfi_param_2,   // c (指针, 64位)
       .param .u32 _Z10vector_addPKfS0_Pfi_param_3    // n (int, 32位)
   )

关键指令逐行分析：

.. list-table:: PTX 指令分析
   :header-rows: 1
   :widths: 40 60

   * - PTX 指令
     - 说明
   * - ``ld.param.u64 %rd1, [param_0]``
     - 加载参数 a (指针) 到寄存器
   * - ``mov.u32 %r3, %ctaid.x``
     - 读取 blockIdx.x → 寄存器 %r3
   * - ``mov.u32 %r4, %ntid.x``
     - 读取 blockDim.x → 寄存器 %r4
   * - ``mov.u32 %r5, %tid.x``
     - 读取 threadIdx.x → 寄存器 %r5
   * - ``mad.lo.s32 %r1, %r3, %r4, %r5``
     - ``idx = blockIdx.x * blockDim.x + threadIdx.x``
   * - ``setp.ge.s32 %p1, %r1, %r2``
     - ``if (idx >= n)`` 设置谓词寄存器
   * - ``@%p1 bra $L__BB0_2``
     - 如果越界，跳转到 ret
   * - ``cvta.to.global.u64 %rd4, %rd1``
     - 将 a 的通用地址转换为全局地址
   * - ``mul.wide.s32 %rd5, %r1, 4``
     - ``idx * sizeof(float)`` = idx * 4
   * - ``add.s64 %rd6, %rd4, %rd5``
     - ``&a[idx]`` = a + offset
   * - ``ld.global.f32 %f1, [%rd8]``
     - 从全局内存加载 b[idx]
   * - ``ld.global.f32 %f2, [%rd6]``
     - 从全局内存加载 a[idx]
   * - ``add.f32 %f3, %f2, %f1``
     - ``c[idx] = a[idx] + b[idx]``
   * - ``st.global.f32 [%rd10], %f3``
     - 存储结果到全局内存
   * - ``ret``
     - 返回

使用的寄存器资源：

- ``%p<2>`` — 2 个谓词寄存器 (predicate)
- ``%f<4>`` — 4 个 32 位浮点寄存器
- ``%r<6>`` — 6 个 32 位整数寄存器
- ``%rd<11>`` — 11 个 64 位整数寄存器

**PTX 的关键特性**：

- **虚拟 ISA**：不绑定具体硬件，同一 PTX 可在不同代 GPU 上运行（驱动 JIT
  编译）
- **无限寄存器**：PTX 层面的寄存器数量是无限的（\ ``.reg .f32 %f<4>`` 只
  是实际使用的数量）
- **显式内存层次**：\ ``ld.global``\ （全局内存）、\ ``ld.shared``\ （共享内
  存）、\ ``ld.local``\ （本地内存）等
- **并行语义**：\ ``%ctaid``\ 、\ ``%ntid``\ 、\ ``%tid`` 分别对应 CUDA
  的内置变量

``vector_add.sm_89.cubin`` — SASS 二进制
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
3.5KB 的 GPU 二进制机器码。PTX 汇编器 ``ptxas`` 针对 sm_89 架构，将虚拟
寄存器映射为物理寄存器，并做指令调度优化。

``vector_add.fatbin.c`` — Fat Binary 嵌入
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``fatbinary`` 将 cubin + PTX 打包后，以 C 源文件形式嵌入。它以 **内联汇
编** 方式定义了一个 ELF 格式的 fat binary：

.. code:: text

   // 文件头标识 (ELF magic)
   .quad 0x00100001ba55ed50   // fat binary header magic

   // ELF header 开始
   .quad 0x41010102464c457f   // "\177ELF" (小端序)
   ...

   // 包含两种 image：
   // 1. kind=elf, sm=89  →  SASS 机器码（直接执行）
   // 2. kind=ptx, sm=89  →  PTX 汇编（JIT 回退或调试）

   // 末尾声明 C 结构体用于运行时注册
   static const __fatBinC_Wrapper_t __fatDeviceText
       __attribute__((aligned(8)))
       __attribute__((section(".nvFatBinSegment"))) =
       { 0x466243b1, 1, fatbinData, 0 };

运行时，CUDA driver 通过 ``.nvFatBinSegment`` 段中的
``__fatDeviceText`` 结构体找到 fat binary，根据实际 GPU 型号选择 ELF
（直接加载执行）或 PTX（JIT 编译）。



完整数据流
----------

.. mermaid:: ../_static/intermediate_dataflow.mmd

关键发现
--------

1. **代码膨胀** — 489 字节的 ``.cu`` 源码经过头文件展开膨胀到 1.2MB+ 的
   预处理文件。CUDA 头文件的体积远大于实际业务代码。

2. **stub 的构造函数机制** — ``__sti____cudaRegisterAll`` 使用
   ``__attribute__((constructor))`` 在 ``main()`` 之前完成 CUDA 运行时初
   始化，用户对此完全无感知。

3. **PTX 是最佳学习材料** — PTX 完整保留了 kernel 的所有逻辑，体积很小
   （1.3KB），比 SASS 更易读，是理解 GPU 编程模型的最佳入口。

4. **Fat Binary 的双保险** — fatbinary 同时保留了 PTX 和 SASS。SASS 用
   于直接执行（零开销），PTX 用于 JIT 回退（兼容不同驱动版本或 JIT 优
   化）。
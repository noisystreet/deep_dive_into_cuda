__cudaRegister* 注册链
==========================

   :doc:`07_host_link` 说明 g++ 如何把 ``.nv_fatbin`` 段链进可执行文件；
   :doc:`../chapter_03_runtime/03_kernel_launch` 说明 ``<<<>>>`` 如何变成
   ``cuLaunchKernel``。两章之间缺的一环是：**main 执行前**，编译器生成的
   constructor 如何把 fat binary 交给 ``libcudart_static``，并在 Runtime 内
   建立 ``host wrapper ↔ device 符号`` 映射。

   本节用 ``objdump`` / ``nm`` / ``readelf`` 对齐 **WPC 单 TU** 与 **RDC 多 TU**
   两条注册路径，并引用 ``crt/host_runtime.h``、``crt/link.stub`` 源码解释宏
   展开。

   环境: CUDA 13.1 / sm_89 / Linux x86-64

.. admonition:: 你知道吗？

   ``__cudaRegisterBinary`` 在源码里看起来像函数调用，实际上是 **宏**——预处理器
   把它展开为 ``__cudaRegisterFatBinary`` + callback + ``__cudaRegisterFatBinaryEnd``
   + ``atexit`` 四步。因此 ``nm vector_add`` 里 **找不到** ``__cudaRegisterBinary``
   符号，但反汇编 ``__sti____cudaRegisterAll`` 能看到完整展开体。

--------------

要回答的三个问题
------------------

1. fat binary 数据（``.nv_fatbin``）如何被 Runtime **发现并加载**？
2. ``vector_add`` 这个 host 函数名如何关联到 ``_Z10vector_addPKfS0_Pfi``？
3. RDC 多 TU 时，为何出现多个 constructor 和 ``__cudaRegisterLinkedBinary_*``？

--------------

复现命令
--------------

.. code:: bash

   cd examples/build && cmake .. && cmake --build . --target vector_add rdc_vector_add
   nm -C vector_add | rg 'Register|fatDevice|device_stub|FatCubin'
   objdump -d vector_add | sed -n '/9329:/,/9385/p'
   readelf -x .nvFatBinSegment vector_add
   readelf -x .init_array vector_add

RDC 对照：

.. code:: bash

   nm -C rdc_vector_add | rg 'RegisterLinked|Prelinked|fatDevice'
   readelf -x .nvFatBinSegment rdc_vector_add
   readelf -x .init_array rdc_vector_add

--------------

编译器侧：谁生成注册代码？
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 36

   * - 文件
     - 角色
   * - ``*.cudafe1.stub.c``
     - cudafe++ 生成：``__device_stub__*``、``__nv_cudaEntityRegisterCallback``、
       ``__sti____cudaRegisterAll``
   * - ``*.fatbin.c``
     - fatbinary 生成：汇编写入 ``.nv_fatbin``，``__fatBinC_Wrapper_t`` 写入
       ``.nvFatBinSegment``
   * - ``crt/link.stub`` + ``*.reg.c``
     - device link 生成：``DEFINE_REGISTER_FUNC``、``__cudaRegisterLinkedBinary`` 内联函数
   * - ``crt/host_runtime.h``
     - ``__cudaRegisterBinary`` 等宏与 ``__cudaLaunch`` 族

``vector_add.cudafe1.stub.c`` 核心片段：

.. code:: c

   static void __nv_cudaEntityRegisterCallback(void **__T7) {
       __nv_dummy_param_ref(__T7);
       __nv_save_fatbinhandle_for_managed_rt(__T7);
       __cudaRegisterEntry(__T7, (void (*)(...))vector_add,
           _Z10vector_addPKfS0_Pfi, (-1));
   }
   static void __sti____cudaRegisterAll(void) {
       __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);
   }

``--keep`` 保留的 stub 仍写 ``__cudaRegisterBinary``；经预处理器展开后，``.cu.o``
里实际是下面要分析的 **FatBinary 三步**。

--------------

``__fatBinC_Wrapper_t``：Runtime 的入口句柄
----------------------------------------------

``fatbinary_section.h`` 定义控制结构：

.. code:: c

   #define FATBINC_MAGIC   0x466243B1
   #define FATBINC_VERSION 1
   #define FATBINC_LINK_VERSION 2

   typedef struct {
     int magic;
     int version;
     const unsigned long long* data;
     void *filename_or_fatbins;
   } __fatBinC_Wrapper_t;

.. list-table::
   :header-rows: 1
   :widths: 12 14 34

   * - version
     - 典型场景
     - 第四字段含义
   * - 1
     - WPC 单 TU / TU 级 relocatable
     - ``0`` 或忽略
   * - 2
     - device link 后的 dlink fatbin
     - 指向 ``__cudaPrelinkedFatbins[]`` 数组

``readelf -x .nvFatBinSegment vector_add`` 实测 WPC 仅 **一个** wrapper：

::

   b1436246 01000000   magic=0x466243b1, version=1
   30a20900 00000000   data → VA 0x9a230 (.nv_fatbin)

RDC 可执行文件 ``rdc_vector_add`` 的 ``.nvFatBinSegment`` 为 **72 B**，含 **3** 个
wrapper：两个 version=1 的 TU relocatable + 一个 version=2 的 dlink 主 fatbin。

--------------

WPC 路径：``__cudaRegisterBinary`` 宏展开
-------------------------------------------

``host_runtime.h`` 中的宏定义：

.. code:: c

   #define __cudaRegisterBinary(X)                                          \
     __cudaFatCubinHandle = __cudaRegisterFatBinary((void*)&__fatDeviceText); \
     { void (*callback_fp)(void **) = (void (*)(void **))(X);               \
       (*callback_fp)(__cudaFatCubinHandle);                                \
       __cudaRegisterFatBinaryEnd(__cudaFatCubinHandle); }                   \
     atexit(__cudaUnregisterBinaryUtil)

``objdump`` 对 ``vector_add`` 中 ``__sti____cudaRegisterAll``（VA ``0x9329``）的还原：

.. code:: text

   lea    __fatDeviceText(%rip), %rdi
   call   __cudaRegisterFatBinary          ; ① 加载 .nv_fatbin
   mov    %rax, __cudaFatCubinHandle       ; 保存 module handle
   lea    __nv_cudaEntityRegisterCallback, %rax
   call   *%rax                            ; ② callback
   mov    __cudaFatCubinHandle, %rdi
   call   __cudaRegisterFatBinaryEnd       ; ③ 结束注册
   lea    __cudaUnregisterBinaryUtil, %rax
   call   atexit                           ; ④ 进程退出时卸载

``.init_array`` 含指向 ``0x9329`` 的指针；glibc 在 ``main`` 之前调用该 constructor。

Callback 内部（``0x92c7``）调用 ``__cudaRegisterFunction``：

.. code:: text

   mov    vector_add, %rsi                 ; host 侧 wrapper 地址
   lea    _Z10vector_addPKfS0_Pfi, %rdx    ; device mangled 名
   mov    __cudaFatCubinHandle, %rdi
   call   __cudaRegisterFunction

``__cudaRegisterFunction`` 实现在 ``libcudart_static.a``（``nm`` 显示为 ``t`` 局部
符号，VA ``0x1dc10``）。它解析 fat binary 内的 cubin/PTX，在 Runtime 模块表中
登记 **device 函数**，并把 host ``vector_add`` 与 device 符号绑定。

Launch 时 ``__device_stub__`` 通过 BSS 中的 ``__handle`` / ``__f`` 查表：

.. code:: text

   nm vector_add | rg device_stub
   __device_stub__Z10vector_add...::__handle
   __device_stub__Z10vector_add...::__f

--------------

Runtime API 分层
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36

   * - 符号
     - 职责
   * - ``__cudaRegisterFatBinary``
     - 读取 ``__fatBinC_Wrapper_t``，映射 ``.nv_fatbin``，返回 handle
   * - ``__cudaRegisterFunction``
     - 在 handle 下注册单个 kernel/device 函数
   * - ``__cudaRegisterFatBinaryEnd``
     - 完成 module 注册，驱动可见
   * - ``__cudaUnregisterFatBinary`` / ``atexit``
     - 进程退出清理
   * - ``__cudaRegisterVar`` / ``__cudaRegisterManagedVar``
     - ``__device__`` 变量、Unified Memory 符号
   * - ``__cudaRegisterLinkedBinary``
     - 内联于 link.stub；收集各 TU 预链接 fatbin，凑齐后触发 dlink 主 fatbin 注册

旧文档中的 ``__cudaRegisterEntry`` 在 callback 源码里仍可见；它同样是宏，展开为
``__cudaRegisterFunction``（见 ``host_runtime.h`` 第 83–84 行）。

--------------

RDC 路径：多 constructor + link.stub
--------------------------------------

:doc:`10_rdc` 从 **段布局** 说明 ``__nv_relfatbin`` 与 ``.nv_fatbin`` 分工；
注册链视角的补充如下。

reg.c 与 link.stub 协作
~~~~~~~~~~~~~~~~~~~~~~~~~~

device link 生成的 ``reg.c``：

.. code:: c

   #define NUM_PRELINKED_OBJECTS 2
   DEFINE_REGISTER_FUNC(_a02a23f4_15_device_utils_cu_20c8e3a2)
   DEFINE_REGISTER_FUNC(_43d618e9_17_vector_add_rdc_cu_052d2be5)

``DEFINE_REGISTER_FUNC`` 宏（``crt/link.stub``）为每个 TU 生成
``__cudaRegisterLinkedBinary_<id>`` 函数，内部调用：

.. code:: c

   __cudaRegisterLinkedBinary(&__fatbinwrap_<id>, callback_fp, ...);

``link.stub`` 中的 ``__cudaRegisterLinkedBinary`` 内联逻辑：

.. code:: c

   __cudaPrelinkedFatbins[__i] = (void*)prelinked_fatbinc->data;
   __callback_array[__i] = callback_fp;
   ++__i;
   if (__i == NUM_PRELINKED_OBJECTS) {
     __cudaFatCubinHandle = __cudaRegisterFatBinary((void*)&__fatDeviceText);
     atexit(__cudaUnregisterBinaryUtil);
     for (__i = 0; __i < NUM_PRELINKED_OBJECTS; ++__i)
       (*__callback_array[__i])(__cudaFatCubinHandle);
     __cudaRegisterFatBinaryEnd(__cudaFatCubinHandle);
   }

要点：

1. 每个 ``.cu`` 的 stub 自带 **独立 constructor**（``readelf .init_array`` 可见
   **两个** 条目：``0x8c94``、``0x9353``），分别注册本 TU 的 relocatable wrapper。
2. 计数达到 ``NUM_PRELINKED_OBJECTS`` 后，才调用 ``__cudaRegisterFatBinary`` 加载
   dlink 主 fatbin：version=2 的 ``__fatDeviceText``，第四字段指向
   ``__cudaPrelinkedFatbins``。
3. 依次调用各 TU 的 callback，对每个 kernel 执行 ``__cudaRegisterFunction``。
4. 最后 ``__cudaRegisterFatBinaryEnd`` 一次。

``nm rdc_vector_add`` 可见完整符号集：

::

   __cudaRegisterLinkedBinary_a02a23f4_15_device_utils_cu_20c8e3a2
   __cudaRegisterLinkedBinary_43d618e9_17_vector_add_rdc_cu_052d2be5
   __cudaRegisterLinkedBinary(...)          ← link.stub 内联体
   __cudaPrelinkedFatbins                   ← BSS 数组
   __fatDeviceText                          ← dlink wrapper, version 2

WPC 与 RDC 注册对比
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 38 38

   * - 维度
     - WPC ``vector_add``
     - RDC ``rdc_vector_add``
   * - constructor 数量
     - 1（``__sti____cudaRegisterAll``）
     - 2（每 TU 一个 ``____cudaRegisterLinkedBinary``）
   * - ``.nvFatBinSegment`` 大小
     - 24 B，1 个 wrapper
     - 72 B，3 个 wrapper
   * - 主 fatbin 段
     - ``.nv_fatbin`` 4064 B
     - ``.nv_fatbin`` 5304 B + ``__nv_relfatbin`` 2832 B
   * - 注册触发
     - 宏 ``__cudaRegisterBinary`` 一次完成
     - ``NUM_PRELINKED_OBJECTS`` 计数后 batch 注册

--------------

从注册到 Launch 的完整时间线
--------------------------------

.. mermaid:: ../_static/register_chain.mmd

.. list-table::
   :header-rows: 1
   :widths: 14 22 44

   * - 阶段
     - 时机
     - 动作
   * - 链接
     - g++ 完成
     - ``.nv_fatbin`` / wrapper 段写入 ELF；``U __cudaRegisterFatBinary`` 待解析
   * - 加载
     - 动态链接器
     - 映射只读 ``.nv_fatbin``；解析 ``libcudart_static`` 符号
   * - 注册
     - ``.init_array``，main 前
     - FatBinary 加载 + ``__cudaRegisterFunction`` 建表
   * - Launch
     - ``main`` 内 ``<<<>>>``
     - stub → ``__cudaGetKernel`` → ``cuLaunchKernel`` → ioctl

多架构 fat binary 的 **image 选择** 发生在 ``__cudaRegisterFatBinary`` 内部——
驱动按当前 GPU CC 挑选 cubin 或 PTX JIT，见 :doc:`../chapter_01_compilation/04_multiarch_fatbinary`。

--------------

与相邻章节的边界
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 32

   * - 主题
     - 章节
   * - fatbin 容器 / ELF 段
     - :doc:`../chapter_01_compilation/03_fatbinary`
   * - g++ 链接与 ``.o`` 段列表
     - :doc:`07_host_link`
   * - RDC 段布局与 reg.c
     - :doc:`10_rdc`
   * - Launch stub 与 ioctl
     - :doc:`../chapter_03_runtime/03_kernel_launch`
   * - libcudart 内部 API 表
     - :doc:`../chapter_03_runtime/02_cudart`
   * - **注册链专篇**
     - **本节**

--------------

*分析基于 CUDA 13.1 (build 37061995)，样例 ``examples/build/vector_add`` 与
``rdc_vector_add`` 的 ``objdump`` / ``readelf`` 输出互证；宏定义引用
``/usr/local/cuda/targets/x86_64-linux/include/crt/host_runtime.h`` 与
``/usr/local/cuda/bin/crt/link.stub``。*

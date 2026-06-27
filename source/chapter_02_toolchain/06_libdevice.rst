libnvvm / libdevice 深度分析
=================================

   第 2.3 节已从架构层面介绍 cicc、libnvvm.so 与 libdevice.10.bc。
   本节聚焦 **libdevice 如何在编译期被链接、优化并消失于最终 PTX**，

.. admonition:: 你知道吗？

   libdevice 提供的数学函数（``sinf``、``expf``、``sqrtf`` 等）
   执行速度因 GPU 架构而异。以 ``sinf`` 为例：它不调用 CPU 的
   libm，而是使用 GPU 硬件指令 ``MUFU.SIN``（MUlti-Function Unit），
   只需 **4 个时钟周期**——比 CPU 上的 ``sinf``（通常 50-100 周期）
   快一个数量级。代价是精度略低：CUDA 的 ``__sinf`` 只保证 ULP
   （unit in the last place）误差在 2 以内，而 IEEE 标准要求 0.5。
   这是 GPU 通过**精度换速度**的典型设计取舍。

   以及 libnvvm API 在其中的角色。

   分析基于 CUDA 13.1 (build 37061995)，主线程序 ``examples/vector_add.cu``，
   对照实验 ``sinf(x)`` kernel。

   环境: Linux x86-64 / sm_89

--------------

libdevice 在工具链中的位置
----------------------------

nvcc 在 verbose 模式下会导出关键路径变量：

::

   #$ NVVMIR_LIBRARY_DIR=/usr/local/cuda/bin/../nvvm/libdevice
   #$ CICC_PATH=/usr/local/cuda/bin/../nvvm/bin

cicc 收到 ``cpp1.ii`` 后，在 NVVM 框架内完成：

1. 将 CUDA C++ device 代码降为 LLVM IR
2. **按需** 将 ``libdevice.10.bc`` 作为额外模块链接进来
3. 运行 NVVM / LLVM Pass（内联、DCE、常量折叠等）
4. 生成 ``*.ptx``

关键发现
-----------

1. **按需链接，而非整体包含** — libdevice.10.bc (454 KB) 只将**实际
   调用的函数**链接到最终 PTX。不调用 ``__nv_sinf`` 的 kernel 不会
   携带任何 libdevice 符号。这与 ``-lcudart_static`` 的静态链接策略
   完全不同。

2. **LLVM Pass 裁剪** — NVVM 的 inlining + DCE Pass 在 libdevice
   链接后立即运行，将数学函数内联展开并消除死代码。最终 PTX 中只
   保留必要的浮点指令序列，libdevice 的 Bitcode 边界完全消失。

3. **标量函数与性能保证** — libdevice 提供的是**标量**数学函数
   （而非向量化版本）。每个线程独立调用，NVVM 的 Pass 只优化
   单线程内代码。这与 Tensor Core 的 warp 级矩阵运算（WMMA）
   形成对比。

4. **454 KB vs 0 B** — ``sinf(x)`` 的 kernel 最终 cubin 仅增加
   几十字节的 SASS 指令，但中间 PTX 包含了从 libdevice 展开的
   完整多项式求值代码。这体现了 Bitcode 链接的"用多少带多少"优势。

--------------

libdevice 文件结构
-------------------

.. code:: text

   路径: /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
   大小: 454,304 字节
   格式: LLVM IR Bitcode (LLVM 7.0.1)


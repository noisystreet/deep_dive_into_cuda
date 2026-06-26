libnvvm / libdevice 深度分析
=================================

   第 2.3 节已从架构层面介绍 cicc、libnvvm.so 与 libdevice.10.bc。
   本节聚焦 **libdevice 如何在编译期被链接、优化并消失于最终 PTX**，
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

.. mermaid:: ../_static/libdevice_link.mmd

这与 CPU 侧的 ``-lm`` 运行时链接完全不同：libdevice 是 **编译期 LLVM 模块**，
不会出现在最终 ELF 的 ``NEEDED`` 列表中。

--------------

libdevice.10.bc 实测概览
---------------------------

文件属性
~~~~~~~~~~

::

   路径: /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
   大小: 464,740 字节 (≈ 454 KB)
   格式: LLVM IR bitcode (file 命令识别)
   目录: CUDA 13.1 仅包含 libdevice.10.bc 一个版本

用 ``llvm-dis`` 反汇编后统计：

-  **352** 个 ``define`` 函数（含内部辅助符号）
-  **349** 个独立的 ``__nv_*`` 入口（去重后）

target triple 为 ``nvptx64-nvidia-gpulibs``——表明它是 **GPU 库 IR**，
供 cicc 在生成 PTX 前链接，而不是 host 侧共享库。

函数命名与分类
~~~~~~~~~~~~~~~~

所有对外符号以 ``__nv_`` 为前缀，与用户可见的 ``sinf`` / ``__sinf`` 等不同层：

+----------------------------+----------------------------------------+
| 前缀 / 模式                | 示例                                   |
+============================+========================================+
| 单精度数学                 | ``__nv_sinf``, ``__nv_cosf``,          |
|                            | ``__nv_expf``, ``__nv_logf``           |
+----------------------------+----------------------------------------+
| 双精度数学                 | ``__nv_sin``, ``__nv_sqrt``            |
+----------------------------+----------------------------------------+
| FMA 变体                   | ``__nv_fmaf_rn/rd/ru/rz``              |
+----------------------------+----------------------------------------+
| 整数 / 位操作              | ``__nv_clz``, ``__nv_popc``            |
+----------------------------+----------------------------------------+
| Warp 归约 (CUDA 9+)        | ``__reduce_add_sync`` 等               |
+----------------------------+----------------------------------------+
| 向量 SIMD                  | ``__nv_vadd2``, ``__nv_vcmpeq4``       |
+----------------------------+----------------------------------------+

``__nv_sinf`` 在 IR 层的实现片段（``llvm-dis`` 输出）：

::

   define float @__nv_sinf(float %a) #0 {
     ...
     %4 = call float @llvm.nvvm.sin.approx.ftz.f(float %a) #6
     ...
     %6 = call float @llvm.nvvm.sin.approx.f(float %a) #6
     ...
   }

可见 libdevice 并非简单转发，而是在 IR 层选择 **NVVM 内建 intrinsic**
（如 ``llvm.nvvm.sin.approx.f``），由后续 Pass 决定是否进一步展开为 PTX 指令序列。

--------------

libnvvm.so：链接与编译 API
------------------------------

cicc 通过 ``dlopen`` 加载 ``libnvvm.so.4.0.0``（≈ 61 MB），
暴露 17 个 ``nvvm*`` 符号。与 libdevice 直接相关的调用链为：

::

   nvvmCreateProgram
     → nvvmAddModuleToProgram      (用户 IR 模块)
     → nvvmAddModuleToProgram      (libdevice.10.bc，按需)
     → nvvmCompileProgram          (-arch=compute_XX 等选项)
     → nvvmGetCompiledResult       (PTX 文本)

``nvvmLazyAddModuleToProgram`` 的存在说明 NVIDIA 也支持 **延迟加载**
libdevice 子集——在大型 TU 中减少初始 IR 体积，但最终仍要在
``nvvmCompileProgram`` 前完成链接。

与 nvlink LTO 的对比
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 28 28

   * - 场景
     - 谁调用 libnvvm
     - libdevice 是否参与
   * - cicc
     - 每个 ``.cu`` 编译单元
     - 是，按引用链接 + 内联
   * - nvlink -lto
     - device 链接器
     - 可能再次参与 LTO 优化

第 2.5 节已讨论 nvlink 的 LTO 路径；本节关注 **单 TU 的 cicc 路径**。

--------------

按需链接：vector_add 不需要 libdevice
----------------------------------------

``examples/vector_add.cu`` 的 kernel 只做向量加法，不调用数学库。
对 ``examples/build/vector_add.ptx`` 检索：

::

   $ rg '\.call|__nv_|__cudart' vector_add.ptx
   (无匹配)

说明 cicc 在链接 libdevice 后，GlobalDCE / 内联分析判定 **无任何符号被引用**，
整库被裁剪，PTX 中不留痕迹。这是 libdevice 以 bitcode 分发而非
``.a`` 静态库的核心优势：**死代码消除在 IR 层完成**。

--------------

实验：sinf 如何进入 PTX
--------------------------

测试 kernel（参数非常量，避免编译期折叠）：

.. code:: cuda

   __global__ void k(float *o, float x) { o[0] = sinf(x); }

编译命令：

.. code:: bash

   nvcc --keep --keep-dir /tmp/sin2_keep -arch=sm_89 -c sin_test2.cu

``sin_test2.ptx`` 特征
~~~~~~~~~~~~~~~~~~~~~~~~

1. **无** ``.call`` 外部函数——libdevice 已被完全内联
2. 出现 ``__cudart_i2opi_f`` 常量数组（π 相关表），用于 argument reduction
3. 大量 ``fma.rn.f32``、分支标签 ``$L__BB0_*``——完整 sin 多项式求值展开
4. PTX 体积从 vector_add 的 ~30 行增至 **170 行**

常量折叠对照
~~~~~~~~~~~~~~

若写成 ``sinf(1.0f)``，cicc 在 IR 层直接算出 ``0.84147096…``，
PTX 仅保留 ``mov.u32 %r1, 1062693540`` + ``st.global.u32``——
连 libdevice 展开都不需要。这解释了为何 **数学库分析必须区分
编译期常量与运行时参数**。

--------------

NVVMIR_LIBRARY_DIR 与版本
----------------------------

nvcc 内部硬编码 ``NVVMIR_LIBRARY_DIR`` 指向 ``nvvm/libdevice``。
用户一般 **不应** 替换该目录；若强行指定旧版 ``libdevice``，
可能与 cicc 内嵌的 NVVM 7.0.1 不兼容，导致 ``nvvmVerifyProgram`` 失败。

文件名中的 ``10`` 表示 **libdevice 主版本**，与 CUDA major 版本对应关系
由 NVIDIA 文档维护；在 13.1 安装树中仅见 ``libdevice.10.bc``。

--------------

与第 2.3 节的关系
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - 主题
     - 所在章节
   * - cicc 体积、Pass 列表
     - :doc:`03_cicc`
   * - libdevice IR 链接与按需裁剪
     - 本节
   * - PTX → cubin
     - :doc:`04_ptxas`
   * - 多 TU device LTO
     - :doc:`05_nvlink`

--------------

*分析基于 CUDA 13.1 (build 37061995)，libdevice 函数计数来自 llvm-dis 输出。*

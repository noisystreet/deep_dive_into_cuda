fatbinary 工具分析
=====================

   ``fatbinary`` 是 CUDA 工具链中的 **Fat Binary 构造器**：把 PTX / ELF cubin
   等 device image 写入自定义容器，并生成可编译进 host 目标文件的
   ``*.fatbin.c``。

   第 1 章 :doc:`../chapter_01_compilation/03_fatbinary` 侧重 **容器字节布局**；
   本节聚焦 **工具本身**——命令行、内部 API、与 ``bin2c`` 的分工，以及 nvcc
   在完整链接路径中如何 **两次** 调用它。

   分析基于 CUDA 13.1 (build 37061995)，样例 ``examples/vector_add.cu``。

   环境: Linux x86-64 / sm_89

--------------

工具定位
------------

在 nvcc 子工具谱系中，``fatbinary`` 体量仅 **1.3 MB**，远小于 cicc / ptxas /
nvlink，但处在 device 编译链的 **最后一环（打包）** 与 host 链接链的
**倒数第二环（dlink 再打包）**：

.. list-table::
   :header-rows: 1
   :widths: 18 12 35

   * - 工具
     - 大小
     - 输入 → 输出
   * - cicc
     - 77 MB
     - CUDA C++ → PTX
   * - ptxas
     - 40 MB
     - PTX → cubin ELF
   * - nvlink
     - 40 MB
     - 多个 device 对象 → 链接 cubin
   * - fatbinary
     - 1.3 MB
     - PTX + cubin → fatbin 容器 + fatbin.c
   * - bin2c
     - 91 KB
     - 任意二进制 → C 数组（通用）

**关键结论**：对 ``vector_add`` 的 ``nvcc --verbose`` 追踪中 **从未出现 bin2c**。
嵌入 host 的 C 源文件由 ``fatbinary --embedded-fatbin`` **直接生成**；
``bin2c`` 是独立小工具，fatbinary 内部已集成同等能力。

--------------

二进制概览
--------------

::

   路径: /usr/local/cuda/bin/fatbinary
   大小: 1,309,648 字节 (≈ 1.3 MB)
   类型: ELF 64-bit LSB pie executable, x86-64, stripped
   版本: Cuda compilation tools, release 13.1, V13.1.115
         Build cuda_13.1.r13.1/compiler.37061995_0

依赖库
~~~~~~~~~~

::

   linux-vdso.so.1
   libpthread.so.0
   libgcc_s.so.1
   libc.so.6

与 ptxas、nvlink 一样，**不依赖** ``libcudart`` / ``libcuda``，纯 host 端离线工具。

内部 API（strings 提取）
~~~~~~~~~~~~~~~~~~~~~~~~~~

``fatbinary`` 二进制内嵌 ``nvFatbin*`` 调用序列，暴露完整打包流水线：

::

   nvFatbinCreate
   nvFatbinAddCubin        ← kind=elf
   nvFatbinAddPTX          ← kind=ptx（附带 ptxasCmdLine）
   nvFatbinAddLTOIR        ← LTO IR（nvlink -lto 路径）
   nvFatbinAddTileIR       ← TileIR
   nvFatbinAddReloc        ← 可重定位 device 对象
   nvFatbinAddIndex
   nvFatbinSize
   nvFatbinGet             ← 输出容器字节
   nvFatbinDestroy

``--image3=kind=...,sm=...,file=...`` 在实现层映射为 ``AddCubin`` / ``AddPTX`` 等调用。
压缩、debug 标记、cmdline 元数据也在此 API 层写入容器头。

--------------

命令行接口
--------------

``fatbinary --help`` 列出 20+ 选项。与 ``vector_add`` 编译最相关的如下：

.. list-table::
   :header-rows: 1
   :widths: 22 38

   * - 选项
     - 作用
   * - ``-64`` / ``-32``
     - 标记 host 指针宽度（64 位 Linux 下恒为 ``-64``）
   * - ``--create <file>``
     - 写出独立 ``*.fatbin`` 容器文件
   * - ``--embedded-fatbin <file>``
     - 生成 ``*.fatbin.c``（含 ``.nv_fatbin`` 段数据 + wrapper）
   * - ``--image3=kind=elf|ptx,sm=NN,file=path``
     - 添加一个 image（nvcc 主路径）
   * - ``--image2=kind=...,file=...``
     - 旧格式；允许多 SM 共存的 host 对象 / 库场景
   * - ``--link``
     - 标记 **device link 后** 的 fatbin（通常仅含链接 cubin）
   * - ``--cicc-cmdline="..."``
     - 记录 device 编译选项元数据（写入容器）
   * - ``--compress [true|false]``
     - 默认 **true**；压缩 PTX / debug image
   * - ``--no-asm``
     - 输出 C 数组而非 inline asm ``.quad`` 序列
   * - ``--device-c``
     - 可重定位 device 代码；配合 ``__nv_relfatbin`` 段

**实测**：``--create`` 与 ``--embedded-fatbin`` 可在 **同一条命令** 中同时使用——
这正是 nvcc verbose 日志中的形式。

--------------

nvcc 如何调用 fatbinary
---------------------------

仅编译（``nvcc -c``）：一次调用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   #$ fatbinary --create="vector_add.fatbin" -64 \
       --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " \
       "--image3=kind=elf,sm=89,file=vector_add.sm_89.cubin" \
       "--image3=kind=ptx,sm=89,file=vector_add.ptx" \
       --embedded-fatbin="vector_add.fatbin.c"

-  **2 个 image**：ELF cubin + PTX 回退
-  **产物**：``vector_add.fatbin`` (4064 B) + ``vector_add.fatbin.c`` (11520 B)
-  **无** ``-link`` 标志
-  **无** nvlink（单 TU ``-c`` 路径不跑 device link）

完整链接（``nvcc -o vector_add vector_add.cu``）：两次调用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: ../_static/fatbinary_tool_flow.mmd

第 1 次 — 与 ``-c`` 相同，打包模块级 PTX + cubin，生成 ``vector_add.o``（含
``.nv_fatbin`` 4064 B）。

第 2 次 — nvlink 之后，**仅** 打包链接 cubin：

::

   #$ nvlink ... -o "vector_add_dlink.sm_89.cubin" ...
   #$ fatbinary --create="vector_add_dlink.fatbin" -64 \
       --cicc-cmdline="..." -link \
       "--image3=kind=elf,sm=89,file=vector_add_dlink.sm_89.cubin" \
       --embedded-fatbin="vector_add_dlink.fatbin.c"

差异对照：

.. list-table::
   :header-rows: 1
   :widths: 14 14 14 14 20

   * - 调用
     - 标志
     - image 数
     - fatbin 大小
     - 嵌入对象
   * - 第 1 次
     - （无 link）
     - 2（PTX+ELF）
     - 4064 B
     - ``vector_add.o``
   * - 第 2 次
     - ``-link``
     - 1（ELF）
     - 1528 B
     - ``vector_add_dlink.o``

``cuobjdump -ptx`` 对 **dlink fatbin** 无输出——链接后 fatbin **不含 PTX
回退**，仅保留最终 SASS cubin。运行时 kernel 注册走 ``link.stub`` 引入的
dlink fatbin（见 :doc:`07_host_link`）。

g++ 最终链接时，两个 ``.o`` 的 ``.nv_fatbin`` 段 **按序合并**：

::

   readelf -SW vector_add | rg nv_fatbin
   [17] .nv_fatbin   ...   000015d8   ← 5592 = 4064 + 1528

``.nvFatBinSegment`` 中出现 **两个** ``__fatBinC_Wrapper_t``（version 1 与 2），
分别指向两段 ``fatbinData``。

--------------

手工复现与验证
----------------

用 ``examples/build/`` 下 ``--keep`` 保留的中间文件，可脱离 nvcc 单独调用：

.. code:: bash

   fatbinary -64 \
     --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " \
     "--image3=kind=elf,sm=89,file=vector_add.sm_89.cubin" \
     "--image3=kind=ptx,sm=89,file=vector_add.ptx" \
     --create=/tmp/manual.fatbin

``cmp`` 对比 ``/tmp/manual.fatbin`` 与 ``vector_add.fatbin`` → **逐字节相同**，
证明 verbose 日志中的参数完整描述了打包行为。

压缩选项
~~~~~~~~~~

默认 ``--compress=true`` 时容器为 **4064 B**；显式 ``--compress=false`` 时增至
**4696 B**（PTX + cubin 明文存储，与 :doc:`../chapter_01_compilation/03_fatbinary`
「明文未压缩」描述在 **关闭压缩** 时一致）。

输出格式：inline asm 与 C 数组
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

默认 ``*.fatbin.c`` 使用 **内联汇编** 写入 ``.nv_fatbin`` 段：

.. code:: c

   asm(
   ".section .nv_fatbin, \"a\"\n"
   ".align 8\n"
   "fatbinData:\n"
   ".quad 0x00100001ba55ed50,...\n"
   ...

``--no-asm`` 则生成 ``static const unsigned long long fatbinData[] = {...}`` 形式，
语义相同，便于非 GCC 工具链或静态分析。

两种模式均包含：

1. ``#include "fatbinary_section.h"``
2. ``__fatBinC_Wrapper_t`` 放入 ``.nvFatBinSegment``（magic ``0x466243B1``）
3. ``fatbinData`` 标签 / 数组放入 ``.nv_fatbin``

--------------

与 bin2c、nvprune 的关系
---------------------------

bin2c
~~~~~

``bin2c`` 仅提供 ``unsigned char name[] = { 0x.., ... }`` 模板，**不** 生成
``__fatBinC_Wrapper_t``、**不** 声明 ELF 段名。fatbinary 输出的 ``*.fatbin.c``
是 **CUDA 专用嵌入格式**，不是 bin2c 的简单套用。

若只需把任意 blob 转为 C 数组（与 CUDA 注册无关），才需要 bin2c。

nvprune
~~~~~~~

``nvprune`` (123 KB) 用于 **裁剪** 已有 fatbin 中 unused 架构 image，属于
发布体积优化工具，不参与编译主路径。与 fatbinary **构造** 互补。

--------------

image2 与 image3
------------------

+--------+---------------------------+----------------------------------------+
| 格式   | 典型场景                  | SM 指定方式                            |
+========+===========================+========================================+
| image3 | nvcc 默认；每 image 独立  | ``sm=89`` 显式写在参数中               |
|        | cubin/ptx 文件            |                                        |
+--------+---------------------------+----------------------------------------+
| image2 | host 静态库 / 预链接对象  | SM 信息已在 cubin 内部                 |
|        |                           |                                        |
+--------+---------------------------+----------------------------------------+

``vector_add`` 走 image3；:doc:`../chapter_01_compilation/04_multiarch_fatbinary`
中多 ``-gencode`` 场景会在 **同一条** fatbinary 命令中重复多个 ``--image3=``。

--------------

与其他章节的边界
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 32

   * - 主题
     - 所在章节
   * - 容器魔数、image 目录、PTX/cubin 偏移
     - :doc:`../chapter_01_compilation/03_fatbinary`
   * - 多架构 ``-gencode`` 与 fatbinary 命令行
     - :doc:`../chapter_01_compilation/04_multiarch_fatbinary`
   * - fatbin 段落入 ELF / g++ 链接
     - :doc:`07_host_link`
   * - nvlink 产出 dlink cubin
     - :doc:`05_nvlink`
   * - fatbinary 工具 CLI 与两次调用
     - **本节**

--------------

*分析基于 CUDA 13.1 (build 37061995)，手工复现与 nvcc --verbose 日志互证。*

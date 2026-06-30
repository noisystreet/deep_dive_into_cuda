前言
===================================

为什么要写这本书
----------------

大多数 CUDA 教程停在 ``cudaMalloc`` 和 ``<<<>>>`` 语法层面——你知道怎么写
kernel，但不一定知道 **编译器把源码变成了什么**，**运行时如何把 kernel 送
上 GPU**，以及 **驱动在内核里做了什么**。

本书从 ``examples/vector_add.cu`` 出发，用一份仅 489 字节的向量加法程序，贯穿以下完整链路：

.. mermaid:: ../_static/arch_overview.mmd

上图展示了五个层次。下面逐层说明各章如何展开分析。

全书结构
--------

第 1 章：编译过程深度分析（6 篇）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

从 ``nvcc --verbose`` 还原 11 步流水线；用 ``--keep`` 观察中间产物膨胀；用
``readelf`` / ``cuobjdump`` 解析 Fat Binary 与多架构 image；在本章末尾衔接
SASS 在 SM 上的执行语义与 PTX JIT 回退。

第 2 章：nvcc 工具链分析（11 篇）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

按编译流水线顺序逐个逆向 nvcc、cudafe++、cicc、ptxas、libdevice、nvlink、
fatbinary 工具、g++ 链接、TileIR、RDC；以 ``__cudaRegister*`` 注册链收束，
把 fat binary 与第 3 章 launch 衔接起来。

第 3 章：运行时与驱动（4 篇）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

用 ``strace`` 跟踪 ``vector_add`` 从 ``cudaMalloc`` 到 ``ioctl(/dev/nvidia0)``，
分析 Driver API、libcudart、kernel launch 与同步机制。

第 4 章：进阶运行时专题（8 篇）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 ``vector_add`` 主线之外，用独立示例分析 Stream/Graph（概念与捕获分两篇
连续阅读）、内存与 UVM、Context、cuModule 动态加载、cuBLAS 库调用等专题。

附录
----

资源推荐与术语表。

推荐阅读顺序
------------

1. **顺序通读** — 第 1→2→3→4 章与编译—加载—执行的时间线一致；第 2 章
   11 注册链是理解第 3 章 3 kernel launch 的前置。
2. **第 1 章 1.5 SASS 执行** — 可先略读，在读完第 2 章 ptxas 与第 3 章
   launch 后回头对照 constant bank 与谓词分歧，体会更深。
3. **第 4 章 Graph** — 先读 Streams/Graphs 概念（4.1），紧接着读 Graph
   捕获专篇（4.2），再进入内存等专题。
4. **专题跳转** — PTX JIT（1.6）与 cuModule 加载（4.3）、注册链（2.11）与
   launch（3.3）可成对对照阅读。

阅读建议
--------

1. **先跑起来** — 在 ``examples/`` 目录执行 ``demo_build.sh``，亲眼看到中
   间产物（\ ``.ptx``\ 、\ ``.cubin``\ 、\ ``.fatbin`` 等）再读对应章节。
2. **跟着工具走** — 每一章引入一类分析视角（编译日志、中间文件、二进制逆
   向、运行时追踪），建议同步在终端复现关键命令。
3. **证据优先** — 文中的大小、符号数量、指令条数均来自实际工具输出，不同
   CUDA 版本或 GPU 架构的具体数值可能有所差异。

环境与约定
----------

- **分析环境**：CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) /
  Linux x86-64
- **示例程序**：``examples/vector_add.cu``（主线）；第 4 章另含
  ``streams_demo``、``graph_capture_demo``、``module_demo`` 等
- **术语定义**：见 :doc:`../appendix/02_glossary`

*Deep Dive Into CUDA — 2026 年 6 月*

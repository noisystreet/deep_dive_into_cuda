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

**第 1 章：编译过程深度分析**

从 ``nvcc --verbose`` 的实际编译日志出发，还原 NVCC 的 11 步 / 5 阶段流水
线；借助 ``--keep`` 保留的中间文件，观察 489 字节源码如何膨胀为 1.2MB+ 的
预处理输出；最后用 ``readelf`` 和 ``cuobjdump`` 解析 Fat Binary 的容器结
构与 SASS 段布局。

**第 2 章：nvcc 工具链分析**

NVCC 本身只是一个 32 MB 的 **Driver Compiler**——它通过 ``execve`` 调度
cudafe++、cicc、ptxas、nvlink 等子工具。本章用 ``file``\ 、\ ``ldd``\ 、
\ ``strings``\ 、\ ``strace`` 等标准工具，逐个逆向分析这些二进制文件的架
构、依赖与职责分工。

**第 3 章：运行时与驱动**

编译产物最终要在 GPU 上运行。本章用 ``strace`` 跟踪 ``vector_add`` 从
``cudaMalloc`` 到 ``ioctl(/dev/nvidia0)`` 的完整调用链，揭示 Runtime API
（libcudart）、Driver API（libcuda.so）与内核模块（nvidia.ko）三层之间的
关系。

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
- **示例程序**：``examples/vector_add.cu``
- **术语定义**：见 :doc:`../appendix/02_glossary`

*Deep Dive Into CUDA — 2026 年 6 月*

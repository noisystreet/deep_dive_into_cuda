资源推荐
=========

本书的分析方法依赖 **官方文档 + 命令行工具 + 系统级追踪** 三类资源。下面按
用途分类列出推荐材料，便于读者在复现实验或深入某一环节时查阅。

官方文档
--------

NVIDIA 官方文档是验证工具行为与 API 语义的第一手来源。本书基于 CUDA 13.1 编
写，不同版本的命令行选项与目录布局可能略有差异，请以对应版本的文档为准。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 资源
     - 说明
   * - `CUDA Toolkit Documentation <https://docs.nvidia.com/cuda/>`__
     - Toolkit 总入口：安装指南、Release Notes、各组件索引
   * - `CUDA C++ Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`__
     - CUDA C++ 语言模型、内存模型、执行模型的权威说明
   * - `NVCC Compiler Driver <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>`__
     - ``nvcc`` 编译选项、两阶段编译、Fat Binary 相关说明
   * - `PTX ISA Reference <https://docs.nvidia.com/cuda/parallel-thread-execution/>`__
     - PTX 指令集参考，阅读 ``.ptx`` 中间产物时的必备手册
   * - `CUDA Driver API <https://docs.nvidia.com/cuda/cuda-driver-api/>`__
     - ``libcuda.so`` 底层接口，对应本书第 3 章 strace 追踪的 Driver 层
   * - `CUDA Runtime API <https://docs.nvidia.com/cuda/cuda-runtime-api/>`__
     - ``libcudart`` 高层接口，对应 Runtime 包装层分析
   * - `CUDA Binary Utilities <https://docs.nvidia.com/cuda/cuda-binary-utilities/>`__
     - ``cuobjdump``\ 、\ ``nvdisasm`` 等二进制分析工具的官方用法

CUDA 工具手册
-------------

本书各章使用的 CUDA 自带工具，可按编译 / 反汇编 / 调试三类查阅：

**编译与链接**

- ``nvcc`` — 编译驱动，调度 cudafe++ / cicc / ptxas / nvlink
- ``ptxas`` — PTX 汇编器，输出 ``.cubin``
- ``nvlink`` — Device 链接器，合并多个编译单元

**二进制分析**

- ``cuobjdump`` — 提取 fat binary 中的 PTX / ELF / SASS 信息
- ``nvdisasm`` — 反汇编 cubin 中的 SASS 指令
- ``fatbinary`` — 打包 PTX 与 cubin 为 fat binary 容器

**运行时调试**

- ``cuda-gdb`` — GPU 内核调试器
- ``compute-sanitizer`` — 内存与竞争检测（原 cuda-memcheck 系列）

上述工具的命令行参考见 `CUDA Binary Utilities <https://docs.nvidia.com/cuda/cuda-binary-utilities/>`__
与 Toolkit 安装目录下的 ``bin/`` 工具 ``--help`` 输出。

系统级分析工具
--------------

本书第 2 章（工具链逆向）与第 3 章（运行时追踪）大量依赖 Linux 标准工具：

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - 工具
     - 本书章节
     - 典型用途
   * - ``strace``
     - 第 3 章
     - 追踪 ``ioctl``\ 、\ ``mmap``\ 、\ ``open`` 等系统调用，还原
       Runtime → Driver → 内核路径
   * - ``file`` / ``readelf``
     - 第 1–2 章
     - 识别 ELF 类型、段表、符号表
   * - ``ldd``
     - 第 2 章
     - 分析 nvcc / cicc / ptxas 等工具的动态库依赖
   * - ``strings`` / ``nm``
     - 第 2 章
     - 提取二进制中的版本字符串、符号名
   * - ``objdump``
     - 第 1 章
     - 反汇编 host 端 ``.o`` / 可执行文件

推荐阅读 `strace(1) <https://man7.org/linux/man-pages/man1/strace.1.html>`__
与 `readelf(1) <https://man7.org/linux/man-pages/man1/readelf.1.html>`__
手册页，理解输出字段含义。

延伸阅读
--------

以下材料有助于建立更完整的背景知识，但不必通读——按需查阅即可。

**GPU 架构与编程**

- `NVIDIA GPU Architecture <https://www.nvidia.com/en-us/data-center/technologies/>`__
  — 各代架构白皮书（Volta、Ampere、Ada 等），理解 SM、Warp、内存层次
- Programming Massively Parallel Processors（Kirk & Hwu）— CUDA 并行编程
  经典教材，侧重算法与性能，而非编译器内部

**编译器与链接**

- Linkers and Loaders（Levine）— ELF 加载、符号解析、重定位机制
- LLVM 官方文档 — 理解 cicc 基于 NVVM/LLVM 的编译框架背景：
  `LLVM Documentation <https://llvm.org/docs/>`__

**Linux 驱动与设备模型**

- Linux Device Drivers（Corbet, Rubini, Kroah-Hartman）— 字符设备、
  ``ioctl`` 与用户态/内核态交互模型
- `Linux PCI <https://docs.kernel.org/PCI/index.html>`__ — 理解 BAR 映射与
  GPU PCIe 枚举（本书第 3 章涉及）

社区与源码
----------

- `CUDA Samples <https://github.com/NVIDIA/cuda-samples>`__ — NVIDIA 官方示
  例，涵盖各 API 用法
- `LLVM NVPTX Backend <https://llvm.org/docs/NVPTXUsage.html>`__ — NVPTX 目标
  后端文档，与 cicc 输出 PTX 的路径相关
- 本书仓库：`noisystreet/deep_dive_into_cuda <https://github.com/noisystreet/deep_dive_into_cuda>`__
  — 文档源文件与 ``examples/vector_add.cu`` 示例

与本书章节的对应关系
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 想了解…
     - 优先查阅
   * - NVCC 编译流水线与中间文件
     - NVCC Compiler Driver + 本书第 1 章
   * - PTX / SASS 指令含义
     - PTX ISA Reference + ``nvdisasm`` 输出对照
   * - nvcc 子工具内部架构
     - 本书第 2 章 + ``strings`` / ``ldd`` 复现
   * - ``cudaLaunchKernel`` 到 ``ioctl`` 的调用链
     - CUDA Driver API + 本书第 3 章 + ``strace``
   * - Fat Binary / cubin 文件格式
     - CUDA Binary Utilities + 本书第 1.3 节

*Deep Dive Into CUDA — 2026 年 6 月*

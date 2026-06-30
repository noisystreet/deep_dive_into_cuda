NVCC 分析：工具链架构
=========================

   对 nvcc 可执行文件进行分析，揭示其作为”驱动编译器 (Driver
   Compiler)“的内部架构

   :doc:`../chapter_01_compilation/01_compilation_pipeline` 已从日志角度画出
   编译地图。本章从 **nvcc 二进制本身** 入手，按 execve 顺序逐个拆解
   cudafe++、cicc、ptxas 等子工具；读完本章末尾 :doc:`11_register_chain`
   后，再进入第 3 章看已注册 fatbin 如何被 launch。

.. admonition:: 你知道吗？

   NVCC 的名字源自 NVIDIA Compiler Collection，与 GCC (GNU Compiler
   Collection) 的命名如出一辙。有趣的是，NVCC 本身并不真正"编译"
   任何代码——它更像一个**编译调度器**：它调用真正的编译器
   （cudafe++ 做语法分析、cicc 生成 PTX、ptxas 做汇编），自己只负责
   参数解析和文件路由。NVIDIA 选择这种"外包"架构而非自研完整编译器，
   是因为可以复用 GCC/LLVM 的 host 编译能力，将精力集中在 GPU 特有
   的前端和后端上。


分析工具与方法
--------------

======================= ========================
工具                    用途
======================= ========================
``file``                检查文件类型
``ldd``                 分析动态链接库依赖
``strings``             提取二进制中的字符串信息
``strace -f -e execve`` 跟踪子进程创建
``readelf``             分析 ELF 结构
文件系统探查            检查 CUDA 工具链目录结构
======================= ========================

--------------

nvcc 二进制概览
------------------

::

   nvcc: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV),
         dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2,
         for GNU/Linux 2.6.32, stripped

============ ===================================================
属性         值
============ ===================================================
**文件大小** 33,371,544 字节 (≈32 MB)
**架构**     x86-64 ELF, PIE, stripped
**链接方式** 动态链接
**编程语言** C++ (大量 std::, template, class 字符串)
**构建版本** ``Build cuda_13.1.r13.1/compiler.37061995_0``
**自述**     ``Cuda compilation tools, release 13.1, V13.1.115``
**自述**     ``Cuda compiler driver``
============ ===================================================

依赖库
~~~~~~~~~~

::

   linux-vdso.so.1
   libpthread.so.0     ← POSIX 线程
   libm.so.6           ← 数学库
   libgcc_s.so.1       ← GCC 运行时
   libc.so.6           ← C 标准库

依赖非常轻量，只有基本的系统库。nvcc 本身并不包含 LLVM/GPU
相关的库——这些在它的子工具中。

文件类型识别
~~~~~~~~~~~~~~~~

nvcc 是一个 **剥离 (stripped)**
的二进制文件（无符号表），这增加了逆向难度。从字符串中提取的关键自我标识：

::

   Cuda compiler driver
   Cuda compilation tools, release 13.1, V13.1.115
   Build cuda_13.1.r13.1/compiler.37061995_0

--------------

子工具链：完整的工具目录
---------------------------

nvcc 不自己编译代码，而是\ **编排 (orchestrate)**
一系列专用子工具。完整的工具链如下：

子工具清单
~~~~~~~~~~~~~~

.. list-table:: 子工具清单
   :header-rows: 1
   :widths: 20 25 15 40

   * - 工具
     - 路径
     - 大小
     - 角色
   * - **nvcc**
     - ``bin/nvcc``
     - **32 MB**
     - 驱动编译器 — 编译流程的总控
   * - **cicc**
     - ``nvvm/bin/cicc``
     - **77 MB**
     - CUDA Device 编译器 — ``.cu`` → PTX
   * - **ptxas**
     - ``bin/ptxas``
     - **40 MB**
     - PTX 汇编器 — PTX → SASS cubin
   * - **nvlink**
     - ``bin/nvlink``
     - **40 MB**
     - Device 链接器 — 链接多编译单元的 device 代码
   * - **cudafe++**
     - ``bin/cudafe++``
     - **15 MB**
     - CUDA 前端 — 解析 CUDA 语法，分离 host/device
   * - **fatbinary**
     - ``bin/fatbinary``
     - **1.3 MB**
     - Fat binary 打包器 — 打包 PTX+cubin 为容器
   * - **bin2c**
     - ``bin/bin2c``
     - **91 KB**
     - 二进制转 C 数组
   * - **nvprune**
     - ``bin/nvprune``
     - **123 KB**
     - Fat binary 裁剪器

相对大小反映了内部复杂度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   cicc      77 MB  ← 基于 LLVM/NVVM 的全功能编译器后端
   ptxas     40 MB  ← PTX → SASS 汇编器，含指令调度和寄存器分配
   nvlink    40 MB  ← Device 代码链接器，处理 ELF cubin
   nvcc      32 MB  ← 驱动编译器核心（调度编排）
   cudafe++  15 MB  ← CUDA 语法解析器
   fatbinary 1.3 MB ← 容器封装工具
   bin2c     91 KB  ← 简单的二进制转 C 工具

所有子工具的依赖
~~~~~~~~~~~~~~~~~~~~

所有子工具 (ptxas, cudafe++, nvlink, fatbinary)
都具有相同的轻量依赖模式：

::

   linux-vdso.so.1
   libpthread.so.0 / libdl.so.2
   libm.so.6
   libgcc_s.so.1
   libc.so.6

没有任何子工具依赖 CUDA 运行时库 (libcudart) 或 CUDA 驱动库
(libcuda)。这证实了它们都是 **纯主机端工具**\ ，仅在 CPU 上执行，不调用
GPU。

--------------

Driver Compiler 架构：strace 证明
------------------------------------

通过 ``strace -f -e execve`` 跟踪 nvcc 编译一个 ``.cu``
文件的完整过程，得到 33 个 execve 调用：

::

   [nvcc]──┬──[sh]──gcc (cc1plus)          ← Step 1: Host 预处理
            ├──[sh]──gcc (cc1plus)          ← Step 2: Device 预处理
            ├──[sh]──cudafe++               ← Step 3: CUDA 语法解析 & 分离
            ├──[sh]──gcc (cc1plus)          ← Step 4: Device 二次预处理
            ├──[sh]──cicc                   ← Step 5: CUDA C++ → PTX
            ├──[sh]──ptxas                  ← Step 6: PTX → SASS cubin
            ├──[sh]──fatbinary              ← Step 7: 打包 fat binary
            ├──[sh]──gcc (cc1plus → as)    ← Step 8: 编译 host 代码 → .o
            ├──[sh]──nvlink                 ← Step 9: Device 代码链接
            ├──[sh]──fatbinary (link)       ← Step 10: 二次打包
            ├──[sh]──gcc (cc1plus → as)    ← Step 11: 编译 link stub → .o
            └──[sh]──g++ (collect2 → ld)   ← Step 12: 最终链接 → 可执行文件

关键发现
~~~~~~~~~~~~

**1. 所有子工具都通过 ``/bin/sh -c`` 启动**

nvcc 并不是直接 ``fork/exec`` 子进程，而是通过 shell 间接启动：

::

   44956 execve("/bin/sh", ["sh", "-c", "--", "ptxas -arch=sm_75 -m64 ..."], ...)

这意味着 nvcc 的命令行参数中可能存在 shell
注入风险（虽然实际中参数全部由 nvcc 自身构造）。

**2. nvcc 本身只执行一个 execve**

nvcc 进程自身在启动后不再调用 execve——它不做编译工作，只是做决策和调度。

**3. 子进程全部是阻塞式串行执行**

从 strace
的时间顺序看，每个子工具必须等前一个完成后才能启动，不存在并行编译。

内部步骤类 (Step Classes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从 nvcc 二进制中提取的内部步骤类，揭示了其面向对象的设计：

================== ==========================================
类名               对应操作
================== ==========================================
``PreprocessStep`` 预处理 ``.cu``/``.c`` 文件
``CompileStep``    编译非 CUDA 文件 (``.c``/``.cc``/``.cpp``)
``CudaFEppStep``   CUDA 前端预处理 → cudafe++
``CiCCStep``       Device 编译 → cicc
``PtxStep``        PTX 汇编 → ptxas
``DevLinkStep``    Device 链接 → nvlink
``ArchiveStep``    创建静态库
``RunStep``        运行阶段（未知）
``NoStep``         无操作
================== ==========================================

--------------

配置机制：nvcc.profile
-------------------------

::

   TOP              = $(_HERE_)/..
   CICC_PATH        = $(TOP)/nvvm/bin
   NVVMIR_LIBRARY_DIR = $(TOP)/nvvm/libdevice
   LD_LIBRARY_PATH += $(TOP)/lib:
   PATH            += $(CICC_PATH):$(_HERE_):
   INCLUDES        += "-I$(TOP)/$(_TARGET_DIR_)/include"
   SYSTEM_INCLUDES += "-isystem" "$(TOP)/$(_TARGET_DIR_)/include/cccl"
   LIBRARIES        = "-L$(TOP)/$(_TARGET_DIR_)/lib/stubs"
                     "-L$(TOP)/$(_TARGET_DIR_)/lib"
   CUDAFE_FLAGS    +=
   PTXAS_FLAGS     +=

这个 profile 文件定义了： - **CICC_PATH**: cicc (device 编译器) 位于
``$TOP/nvvm/bin/`` - **NVVMIR_LIBRARY_DIR**: NVVM IR 库位于
``$TOP/nvvm/libdevice/`` - **INCLUDES**: CUDA 头文件路径 -
**LIBRARIES**: CUDA 库文件路径

--------------

nvcc 内部架构推测
--------------------

整体架构 (基于字符串分析)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid:: ../_static/nvcc_architecture.mmd

编译阶段决策流程
~~~~~~~~~~~~~~~~~~~~

.. mermaid:: ../_static/phase_decision.mmd

文件扩展名路由
~~~~~~~~~~~~~~~~~~

nvcc 内部的字符串显示它根据不同的文件扩展名分配不同的编译路径：

+--------------+----------------------+--------------------------+
| 文件类型     | 扩展名               | 编译路径                 |
+==============+======================+==========================+
| CUDA 源文件  | ``.cu``              | preprocess → cuda        |
|              |                      | frontend → cicc → PTX →  |
|              |                      | …                        |
+--------------+----------------------+--------------------------+
| GPU 中间文件 | ``.gpu``             | cicc compile into cubin  |
+--------------+----------------------+--------------------------+
| PTX 文件     | ``.ptx``             | ptxas → cubin            |
+--------------+----------------------+--------------------------+
| Cubin 文件   | ``.cubin``           | 直接链接                 |
+--------------+----------------------+--------------------------+
| C/C++ 源文件 | ``.c/.cc/.cpp/.cxx`` | preprocess → compile →   |
|              |                      | link                     |
+--------------+----------------------+--------------------------+
| 目标文件     | ``.o``               | link                     |
+--------------+----------------------+--------------------------+

内部工具路径构造
~~~~~~~~~~~~~~~~~~~~

nvcc 从二进制内部的字符串模板构造子工具路径：

::

   "$CICC_PATH/cicc"          → 实际路径: /usr/local/cuda/nvvm/bin/cicc
   "%CICC_PATH%/cicc"         → Windows 变体
   "$CICC_PATH/cicc.alt"      → 备选 device 编译器
   "/crt/link.stub"           → 链接 stub 模板
   "/crt/prelink.stub"        → 预链接 stub 模板

--------------

关键设计思想
---------------

Driver Compiler 模式
~~~~~~~~~~~~~~~~~~~~~~~~

nvcc 不是一个”编译器”，而是一个
**编译驱动**\ 。它的职责不是生成代码，而是：

1. **解析** 用户输入的命令行参数
2. **确定** 需要执行的编译阶段序列
3. **构造** 每个子工具的命令行
4. **调度** 子工具的执行并传递中间文件
5. **清理** 临时文件

这类似于 GCC 的 ``gcc`` 驱动相对于内部的 ``cc1`` / ``cc1plus``\ 。

为什么是独立的子工具？
~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| 原因                              | 说明                              |
+===================================+===================================+
| **模块化**                        | 每                                |
|                                   | 个子工具负责单一任务，可独立更新  |
+-----------------------------------+-----------------------------------+
| **Licensing**                     | cicc 基于 LLVM/NVVM，cudafe++ 是  |
|                                   | NVIDIA                            |
|                                   | 专                                |
|                                   | 有，分离可以保持清晰的许可证边界  |
+-----------------------------------+-----------------------------------+
| **专业优化**                      | ptxas 的 SASS                     |
|                                   | 代码生成                          |
|                                   | 需要深度硬件知识，与语法解析无关  |
+-----------------------------------+-----------------------------------+
| **并行开发**                      | 各团队可独立开发                  |
|                                   | cicc、ptxas、cudafe++             |
+-----------------------------------+-----------------------------------+
| **选择性部署**                    | 某些场                            |
|                                   | 景可只部署部分工具（如嵌入式只需  |
|                                   | ptxas + fatbinary）               |
+-----------------------------------+-----------------------------------+

两种编译路径的对比
~~~~~~~~~~~~~~~~~~~~~~

nvcc 管理的两条编译路径使用了完全不同的技术栈：

============ ==================== ========================
维度         Host 路径            Device 路径
============ ==================== ========================
**前端**     gcc/g++ (系统编译器) cudafe++ (NVIDIA 专有)
**中端**     gcc 优化             cicc (基于 LLVM/NVVM)
**后端**     gcc 代码生成         ptxas (专有 SASS 汇编器)
**链接器**   g++ / collect2 / ld  nvlink (专有 ELF 链接器)
**目标格式** 标准 ELF (x86-64)    特殊 ELF (NVIDIA CUDA)
============ ==================== ========================

profiler 配置的动态性
~~~~~~~~~~~~~~~~~~~~~~~~~

``nvcc.profile`` 的路径变量通过环境变量动态构造： - ``$(_HERE_)`` = nvcc
自身所在目录 - ``$(_TARGET_DIR_)`` = ``targets/x86_64-linux`` -
``$(_TARGET_SIZE_)`` = 空 (64位) 或 ``_32``

这种设计允许同一个 nvcc 二进制支持不同的目标平台（x86-64, arm64,
etc.）。

--------------

总结
-------

nvcc 是一个 **32MB 的 C++
驱动编译器**\ ，它本身不产生任何代码，而是编排了一个由 6
个主要子工具组成的工具链：

::

   nvcc (Driver)
     ├── gcc/g++ (Host 编译器已有工具)
     ├── cudafe++ (CUDA 语法解析，15 MB)
     ├── cicc (CUDA→PTX 编译器，77 MB，基于 LLVM/NVVM)
     ├── ptxas (PTX→SASS 汇编器，40 MB)
     ├── fatbinary (打包嵌入，1.3 MB)
     └── nvlink (Device 链接器，40 MB)

通过 ``strace`` 验证，编译一个 ``.cu`` 文件共触发 **33 次
execve**\ ，涉及 **12 个阶段**\ 、\ **6 种不同的专用工具**\ 、\ **3 次
gcc 调用** 和 **2 次 fatbinary 调用**\ 。

这种架构设计体现了\ **关注点分离**\ 原则：每个子工具专注解决一个明确定义的问题，而
nvcc 作为”指挥家”确保整个流程有序执行。

--------------

*分析基于 CUDA 13.1 (build 37061995)，不同版本可能有所差异。*

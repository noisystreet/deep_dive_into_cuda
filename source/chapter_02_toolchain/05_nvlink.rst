NVLINK 分析：CUDA Device 链接器
===================================

   nvlink = NVIDIA Linker，是 CUDA 工具链中的 device 代码链接器，
   负责将多个编译单元 (.o) 中的 device 代码链接为单一的 device
   可执行镜像

   分析基于 CUDA 13.1 (build 37061995)

--------------

二进制概览
-------------

基础信息
~~~~~~~~~~~~

::

   文件: /usr/local/cuda/bin/nvlink
   大小: 41,776,688 字节 (≈ 40 MB)
   类型: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV),
         dynamically linked, stripped
   版本: Build cuda_13.1.r13.1/compiler.37061995_0

依赖库
~~~~~~~~~~

::

   linux-vdso.so.1
   libpthread.so.0     ← 多线程支持
   libdl.so.2          ← dlopen/dlsym（加载 libnvvm.so、libtileiras.so）
   libm.so.6           ← 数学库
   libgcc_s.so.1
   libc.so.6

**关键发现**: nvlink 通过 ``dlopen`` 动态加载关键功能库，包括： -
``libnvvm.so`` — LTO 模式下调用 NVVM 编译器进行链接时优化 -
``libtileiras.so`` — TileIR JIT 编译（91 MB 的大库）

与 GNU ld 的宏观对比
~~~~~~~~~~~~~~~~~~~~~~~~

======== ====================== ==========================
功能     GNU ld / gold          nvlink
======== ====================== ==========================
输入     .o (ELF), .a (archive) .o (cubin ELF), .a, .ptx
输出     可执行文件 / .so       cubin (device 可执行)
符号解析 全局符号表合并         device 符号解析
重定位   地址修正               设备地址重定位
LTO      LTO + plugin           NVVM LTO + dlopen libnvvm
库搜索   ``-l -L``              ``-l -L``
增量链接 ``-r``                 ``-r`` (–relocatable-link)
======== ====================== ==========================

--------------

nvlink 完整命令行选项
------------------------

nvlink 支持 50+ 选项，是所有子工具中功能最丰富的：

基本选项
~~~~~~~~~~~~

======================== ============= ============================
选项                     别名          说明
======================== ============= ============================
``--output-file <file>`` ``-o``        输出文件
``--arch <gpu>``         ``-arch``     目标 GPU 架构
``--cpu-arch <cpu>``     ``-cpu-arch`` CPU 目标架构 (默认: unknown)
``--machine <bits>``     ``-m``        64 位模式
``--verbose``            ``-v``        打印统计信息
``--version``            ``-V``        版本信息
======================== ============= ============================

库和输入
~~~~~~~~~~~~

================================== ===============================
选项                               说明
================================== ===============================
``--library <lib>`` (``-l``)       指定链接库 (如 ``-lcudadevrt``)
``--library-path <path>`` (``-L``) 库搜索路径
``--input-file <file>``            输入文件
================================== ===============================

链接控制
~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| 选项                              | 说明                              |
+===================================+===================================+
| ``--relocatable-link`` (``-r``)   | 可重定位                          |
|                                   | /增量链接（不生成最终可执行文件） |
+-----------------------------------+-----------------------------------+
| ``                                | 生成 cudaRegister 例程的输出文件  |
| --register-link-binaries <file>`` |                                   |
+-----------------------------------+-----------------------------------+
| ``--preserve-relocs``             | 保留已解析的重定位信息            |
+-----------------------------------+-----------------------------------+
| ``--keep-system-libraries``       | 不优化掉系统库代码 (cudadevrt)    |
+-----------------------------------+-----------------------------------+
| ``--kernels-used <kernel>,...``   | 指定使用的 kernel，其余视为死代码 |
+-----------------------------------+-----------------------------------+
| ``                                | 指定使用的变量，其余视为死代码    |
| --variables-used <variable>,...`` |                                   |
+-----------------------------------+-----------------------------------+
| ``--use-host-info``               | 使用 host 引用信息移除未使用的    |
|                                   | device 代码                       |
+-----------------------------------+-----------------------------------+
| ``--ignore-host-info``            | 忽略 host 引用信息                |
+-----------------------------------+-----------------------------------+

LTO (链接时优化)
~~~~~~~~~~~~~~~~~~~~

================================ ====================
选项                             说明
================================ ====================
``--link-time-opt`` (``-lto``)   启用链接时优化
``--dlto``                       同 ``-lto``
``--nvvmpath <path>``            libnvvm 库路径
``--split-compile <N>``          NVVM 分片编译线程数
``--split-compile-extended <N>`` 扩展分片编译线程数
``--emit-ptx``                   LTO 时输出 PTX
``--maxrregcount <N>``           LTO 时的最大寄存器数
``--Xptxas <opts>``              透传 ptxas 选项
``--Xnvvvm <opts>``              透传 NVVM 选项
================================ ====================

Device Link 相关
~~~~~~~~~~~~~~~~~~~~

=================================== ====================
选项                                说明
=================================== ====================
``--debug`` (``-g``)                debug 编译
``--device-stack-protector``        启用栈保护
``--suppress-debug-info``           不保留调试信息
``--suppress-stack-size-warning``   抑制栈大小警告
``--dump-callgraph``                打印调用图
``--report-arch``                   在错误信息中报告架构
``--gen-host-linker-script <type>`` 生成 host 链接脚本
=================================== ====================

支持的目标架构
~~~~~~~~~~~~~~~~~~

::

   sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_90a,
   sm_100, sm_100a, sm_100f,
   sm_103, sm_103a, sm_103f,
   sm_110, sm_110a, sm_110f,
   sm_120, sm_120a, sm_120f,
   sm_121, sm_121a, sm_121f

   对应 compute_* 虚拟架构
   以及 lto_* 变体 (LTO 中间表示)

CPU 架构支持:

::

   X86_64, X86, AARCH64, ARMv7, PPC64LE, unknown

--------------

nvlink 核心链接过程
----------------------

链接流程总览
~~~~~~~~~~~~~~~~

.. mermaid:: ../_static/nvlink_flow.mmd

链接类型
~~~~~~~~~~~~

完整链接 (默认)
^^^^^^^^^^^^^^^

多编译单元的 device 代码合并为一个 **可执行 cubin**\ ，含完整的
``.text`` 和 ``.nv.info``\ 。

可重定位链接 (``-r``)
^^^^^^^^^^^^^^^^^^^^^

生成增量链接的中间对象，保留重定位信息用于后续链接。类似于 GNU ld 的
``-r``\ 。

LTO 链接 (``-dlto``)
^^^^^^^^^^^^^^^^^^^^

最复杂的模式。输入中可以包含 LLVM IR bitcode（lto\_\* 架构），nvlink
在链接时： 1. 调用 ``dlopen("libnvvm.so")`` 加载 NVVM 编译器 2. 将所有
LTO IR 模块合并为一个 3.
执行跨模块优化（内联、全局常量传播、死函数消除） 4. 调用 cicc (通过
libnvvm) 重新生成 PTX 5. 调用 ptxas (JIT compile) 重新生成 SASS

::

   字符串证据:
     "do link-time optimization, alias for -lto"
     "force doing partial LTO when -dlto"
     "force doing whole program LTO when -dlto"
     "nvvm options (only used with LTO)"
     "Emit ptx file if LTO is used."
     "ciC-lto"
     "error in LTO callback"

符号解析规则
~~~~~~~~~~~~~~~~

从字符串提取的符号处理逻辑：

::

   "adding global symbols of same name"     ← 同名列添加
   "alias to unknown symbol"                 ← 别名指向未知符号
   "Allow undefined globals and their relocations" ← 允许未定义符号
   "Cannot alias a function to itself"       ← 函数不能别名为自身
   "common symbol"                            ← common 符号处理
   "Common symbol '%s' in file '%s' has larger size" ← common 符号大小检查
   "Could not replace weak symbol '%s'"      ← 弱符号替换失败

nvlink 实现了完整的 ELF 链接语义，包括：

=========================== =======================================
符号类型                    nvlink 处理
=========================== =======================================
**全局强符号 (STB_GLOBAL)** 唯一定义，多重定义报错
**全局弱符号 (STB_WEAK)**   可被强符号覆盖
**Common 符号**             取最大 size
**局部符号 (STB_LOCAL)**    按编译单元隔离
**未定义符号**              允许 (通过 ``Allow undefined globals``)
=========================== =======================================

重定位机制
~~~~~~~~~~~~~~

从字符串确认识别的重定位类型：

::

   EIVALUE_SYM_KIND_PCREL   ← PC 相对重定位
   R_CUDA_*                 ← CUDA 专有重定位类型

nvlink 处理的重定位类型不同于标准 ELF (x86-64), 而是 **CUDA
专用的重定位类型** (``R_CUDA_*``)，包括：

-  **PC 相对重定位**: 分支指令 (BRA, CALL) 的目标地址修正
-  **绝对重定位**: 常量 bank 索引、全局地址修正
-  **常量 bank 重定位**: ``.nv.constant`` 段中的 kernel 参数地址

--------------

LTO 与 JIT 编译
------------------

nvlink 内部集成了一个完整的 **JIT 编译架构**\ ：

LTO 模式调用链
~~~~~~~~~~~~~~~~~~

::

   nvlink --dlto -arch=sm_89 input.o
           │
           ├── dlopen("libnvvm.so")
           │   └── nvvmAddModuleToProgram() ← 加载 LLVM IR
           │   └── nvvmCompileProgram()     ← 跨模块优化
           │   └── 生成优化后的 PTX
           │
           ├── fork/exec ptxas             ← PTX → SASS
           │
           └── 合并 SASS 输出

TileIR JIT
~~~~~~~~~~~~~~

::

   "Can't JIT TileIR without libtileiras"
   "Can't JIT TileIR with this driver or tool"
   "elfLink JIT compile failed"
   "elfLink JIT link failed"
   "FNLZR: Starting JIT"
   "FNLZR: Ending JIT"

nvlink 支持 **TileIR**——一种用于 Tensor Core/McC 的高级中间表示。TileIR
的 JIT 编译引擎包含在 ``libtileiras.so``\ （91 MB）中。

LTO 架构支持
~~~~~~~~~~~~~~~~

nvlink 支持特殊的 ``lto_*`` 架构，用于 LTO 输入：

::

   lto_75, lto_80, lto_86, lto_87, lto_88, lto_89, lto_90, lto_90a,
   lto_100, lto_100a, lto_100f,
   lto_103, lto_103a, lto_103f,
   lto_110, lto_110a, lto_110f,
   lto_120, lto_120a, lto_120f,
   lto_121, lto_121a, lto_121f

这些 ``lto_*`` 架构对应的输入是 NVVM IR bitcode，而非 SASS cubin。

--------------

libcudadevrt.a — CUDA Device Runtime
---------------------------------------

虽然不属于 nvlink 本身，但 libcudadevrt 是 nvlink
链接过程中最重要的输入库：

::

   文件: /usr/local/cuda/lib64/libcudadevrt.a
   大小: 1,023,580 字节 (≈ 1 MB)
   类型: ar archive
   内容: cuda_device_runtime.o  (单个目标文件)

作用
~~~~~~~~

libcudadevrt 提供了 **device 端运行时支持**\ ，包括： -
``__cudaLaunchKernel`` (device 端动态 kernel 启动) -
``__cudaPushCallConfiguration`` (设置 kernel 启动参数) - Device
端内存管理 (``cudaMalloc``/``cudaFree`` 的 device 端入口) - 原子操作包装

在 nvlink 命令行中使用 ``-lcudadevrt`` 链接。

与 nvlink 的关系
~~~~~~~~~~~~~~~~~~~~

::

   nvlink -arch=sm_89 a.o b.o -lcudadevrt -o linked.cubin
                                     │
                                     ▼
                             libcudadevrt.a
                             └── cuda_device_runtime.o
                                 ├── device 端 kernel launch 支持
                                 ├── device 端内存操作
                                 └── 运行时基础设施

--------------

从 nvlink 生成的可执行文件
-----------------------------

以最终 ``vector_add`` 可执行文件为例，nvlink 生成了关键输出：

::

   nvlink 的输出:
     _dlink.sm_89.cubin           ← 链接后的 device 可执行代码
     _dlink.reg.c                 ← cudaRegister 例程（运行时注册用）

   后续处理:
     fatbinary -link → 打包为 fat binary
     gcc (link.stub) → 编译 register stub
     g++ → 最终链接

register stub 机制
~~~~~~~~~~~~~~~~~~~~~~

nvlink 生成的 ``--register-link-binaries`` 文件是一个 C
源文件，包含了在每个链接的 cubin 上调用 ``cudaRegisterBinary``
的代码。这使得 CUDA 运行时可以在程序启动时自动加载 device 代码。

--------------

Fat Binary 嵌入与弹性架构
----------------------------

多种输入类型处理
~~~~~~~~~~~~~~~~~~~~

nvlink 能够处理不同类型的 device 代码，并通过不同的路径处理：

::

   ELF cubin (.o)     → 直接提取 .text SASS 代码
   PTX (.ptx)         → 调用 ptxas JIT 编译为 SASS
   LTO IR (lto_*)     → 调用 libnvvm 优化 + ptxas 编译
   Fat binary (.fatbin) → 解包后提取 cubin
   Archive (.a)       → 解包后递归处理

架构匹配与转换
~~~~~~~~~~~~~~~~~~

::

   "Changed architecture of malformed elf candidates"
   "Changed architecture of non-elf candidates"
   "Cubin profile '%s' cannot be virtual"
   "Suppress warning when object does not contain code for target arch"

nvlink 能够处理输入文件架构与目标架构不匹配的情况——通过 JIT 编译（PTX +
ptxas）重新生成目标架构的代码。

--------------

nvlink 与 GNU ld 的关键差异
------------------------------

============== ================= =============================
维度           GNU ld            nvlink
============== ================= =============================
**输入格式**   标准 ELF + DWARF  特殊 CUDA ELF + NVVM IR + PTX
**输出格式**   x86-64 ELF        CUDA cubin
**重定位类型** R_X86_64\_\*      R_CUDA\_\*
**段合并**     .text/.data/.bss  .text + .nv.\*
**LTO**        LTO plugin (LLVM) libnvvm (dlopen)
**符号表**     标准符号          device 函数符号 (mangled)
**库搜索**     -l -L             -l -L
**寄存器分配** N/A               通过 ptxas LTO 控制
============== ================= =============================

--------------

内部架构总结
---------------

::

   nvlink (40 MB)
           │
           ├── ELF 解析器
           │   ├── 读取 cubin ELF header/sections
           │   ├── 校验格式正确性
           │   └── 提取 section 数据
           │
           ├── 符号解析器
           │   ├── 构建全局符号表
           │   ├── 强弱符号解析
           │   └── 多重定义检测
           │
           ├── LTO 引擎 (可选)
           │   ├── dlopen libnvvm.so
           │   ├── 合并 NVVM IR
           │   ├── 跨模块优化
           │   └── JIT 编译 (cicc + ptxas)
           │
           ├── TileIR JIT (可选)
           │   ├── dlopen libtileiras.so (91 MB)
           │   └── Tensor Core 代码生成
           │
           ├── 段合并器
           │   └── .text + .nv.* sections
           │
           ├── 重定位器
           │   ├── R_CUDA_* 重定位处理
           │   └── 常量 bank 修正
           │
           └── 输出生成器
               ├── 生成 ELF cubin
               ├── 生成 register stub (.reg.c)
               └── 调用 fatbinary

nvlink vs 其他链接器的大小对比
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============ ======= ================
链接器       大小    目标
============ ======= ================
**nvlink**   40 MB   CUDA device code
**ld (GNU)** ~1.5 MB 通用 ELF
**lld**      ~3 MB   通用 ELF
**gold**     ~1.5 MB 通用 ELF
============ ======= ================

nvlink 是通用链接器的 **30-40 倍大小**\ ，因为它集成了 LTO 引擎、JIT
编译器和 ELF 处理。

--------------

关键发现
------------

1. **dlopen 架构**: nvlink 通过 ``dlopen`` 动态加载 ``libnvvm.so`` 和
   ``libtileiras.so``\ ，按需启用 LTO 和 Tensor Core JIT 编译。

2. **双重 JIT 能力**: 同时支持 NVVM IR (LTO) 和 TileIR 两种高级 IR 的
   JIT 编译。

3. **完整的 ELF 链接器**: nvlink 实现了与 GNU ld
   等价的功能——符号解析、强弱符号、common
   符号、重定位、段合并、增量链接。

4. **CUDA 专用重定位类型**: 使用 ``R_CUDA_*`` 而非标准
   ``R_X86_64_*``\ ，针对 GPU 指令编码优化。

5. **Register stub 生成**: nvlink 生成 ``cudaRegisterBinary``
   代码确保运行时自动加载 device 代码。

6. **与 ptxas 共享代码**: nvlink 二进制中包含与 ptxas 相同的
   ``AdvancedPhase*`` 枚举和优化逻辑，说明两者共享同一代码库。

--------------

*分析基于 CUDA 13.1 (build 37061995)*

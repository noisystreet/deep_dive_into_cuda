TileIR 与 tileiras 分析
==========================

   CUDA 13.1 引入 CUDA Tile 编程模型：用 Tile IR 字节码描述
   「分块（tile）」级计算，由 tileiras 汇编为 SASS，并可嵌入 fat binary。
   这与传统 SIMT 路径（cicc → PTX → ptxas）并行存在。

   本节梳理 TileIR 在工具链中的入口、库分工与 JIT 路径。容器内 TileIR
   image 的格式细节仍见 :doc:`../chapter_01_compilation/03_fatbinary` 与
   :doc:`08_fatbinary`；:doc:`05_nvlink` 中曾以 libtileiras 泛指 JIT
   引擎——实测在本环境（Driver 595.58 + CTK 13.1）中，链接期实际
   dlopen 的是 libnvidia-tileiras.so（驱动侧安装包）。

   环境: CUDA 13.1 / Driver 595.58.03 / Linux x86-64

   样例 examples/vector_add.cu 走经典 SIMT 路径，不含 TileIR——
   本章用工具链二进制分析与 tileiras 命令行互证。

--------------

Tile IR 是什么
----------------

NVIDIA Tile IR 规范将其定义为：

   一种可移植的低层 tile 虚拟机与指令集，把 GPU 建模为基于 tile 的处理器，
   而非传统 SIMT 线程模型。

与 PTX 的对照：

.. list-table::
   :header-rows: 1
   :widths: 16 18 18 28

   * - 维度
     - PTX / NVVM 栈
     - Tile IR 栈
     - 典型入口
   * - 中间表示
     - PTX 文本 / NVVM bitcode
     - Tile IR 字节码
     - cuTile、框架前端、cicc
   * - 汇编器
     - ptxas
     - tileiras
     - bin/tileiras
   * - 目标架构（13.1）
     - sm_50 … sm_89 等
     - sm_100 起（Blackwell 类）
     - tileiras --gpu-name=sm_100
   * - vector_add
     - 是
     - 否
     - 仅 SIMT

官方文档：`Tile IR 规范 <https://docs.nvidia.com/cuda/tile-ir/latest/index.html>`__

--------------

两个 tileiras 组件（不要混淆）
--------------------------------

实测存在两个不同物理文件，职责重叠但安装位置不同：

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 36

   * - 组件
     - 路径
     - 大小
     - 角色
   * - tileiras CLI
     - ``/usr/local/cuda/bin/tileiras``
     - 91,451,032 B
     - 离线「Tile IR optimizing assembler」
   * - libnvidia-tileiras
     - ``/usr/lib/x86_64-linux-gnu/libnvidia-tileiras.so.*``
     - 97,276,648 B
     - nvlink 链接期 JIT 通过 dlopen 加载

CTK 的 nvvm/lib64/ 下没有独立的 libtileiras.so（仅有
libnvvm.so）。nvlink 字符串中的

::

   Can't JIT TileIR without libtileiras

是逻辑名；实际解析的 soname 模式为 libnvidia-tileiras.so.590.56
（主版本随驱动变化，本机为 …595.58.03）。

nm -D libnvidia-tileiras.so 导出与 nvlink/fatbinary 字符串一致的
nvTileIR API——说明 CLI 与驱动库共享同一套 TileIR 编译接口。

--------------

tileiras 命令行工具
-----------------------

概览
~~~~~~~~

::

   $ file /usr/local/cuda/bin/tileiras
   ELF 64-bit LSB executable, stripped

   $ tileiras --version
   Cuda compilation tools, release 13.1, V13.1.80
   Build local.local.36836380_

依赖：libpthread、librt、libdl、libm、libc——与 ptxas
同类，纯 host 工具，不链接 libcudart。

用法（``tileiras --help`` 摘要）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   USAGE: tileiras [options] <tile bytecode file>

   --gpu-name=sm_100 | sm_103 | sm_110 | sm_120 | sm_121
   --opt-level=<N>          默认 3
   --output-file=<file>     输出 cubin
   --device-debug / -g
   --lineinfo
   --host-arch / --host-os  生成 host 侧辅助信息时使用

输入必须是 Tile IR 字节码；对普通文本文件：

::

   $ tileiras /tmp/fake.bc -o /tmp/out.cubin --gpu-name=sm_100
   error: input does not correspond to Tile IR bytecode

字符串中还可见 ``it looks like MLIR bytecode instead``——说明 Tile IR 与
MLIR 字节码在魔数层可区分。

与 SIMT 工具链的对比
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 22

   * -
     - ptxas
     - tileiras
   * - 输入
     - PTX 文本
     - Tile IR 字节码
   * - 典型前置
     - cicc
     - cuTile / cicc Tile 前端
   * - 架构 flag
     - -arch sm_89
     - --gpu-name=sm_100
   * - CTK 13.1 体积
     - 40 MB
     - 91 MB

--------------

nvTileIR 编译 API
-------------------

``strings nvlink``、``strings fatbinary``、``nm libnvidia-tileiras`` 共同
暴露的 C API 序列：

::

   nvTileIRIsValid
   nvTileIRCreateAssembleConfig
   nvTileIRDestroyAssembleConfig
   nvTileIRAssembleConfigSetPTXCompiler    ← 内部仍会调用 ptxas
   nvTileIRCreateProgram
   nvTileIRDestroyProgram
   nvTileIRCompileProgram
   nvTileIRGetCompiledProgram
   nvTileIRPickVersion
   nvTileIRIsCompatible
   nvTileIRGetVersion / GetMinArch / GetSize

**设计含义**：TileIR 最终仍要落到 SASS；``SetPTXCompiler`` 表明 tileiras
与 **ptxas / libnvvm** 栈存在耦合，而非完全独立的第三条机器码后端。

--------------

工具链入口（实测归纳）
------------------------

.. mermaid:: ../_static/tileir_entry.mmd

入口 1：cicc 与 ``__tile__`` 源码
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``strings cicc`` 显示 CUDA 13.1 device 编译器已感知 tile 语义：

::

   __device__, __global__, __tile__ or __tile_global__
   __tile_builtin__
   libTileIRCompiler_shared.so
   failed tileir gen!
   unable to lookup tile handler function!

说明 cicc 在特定源码路径下会 ``dlopen`` **TileIR 编译器共享库**（本机
CTK 树中 **未** 以独立 ``.so`` 文件安装，可能内嵌于编译器包或由 cuTile
分发）。``vector_add.cu`` 不含 ``__tile__``，不会触发该路径。

入口 2：nvcc 调度
~~~~~~~~~~~~~~~~~~~~

``strings nvcc`` 中的相关符号：

::

   tileiras
   tilebc
   TilebcTarget
   --tile-only

表明 nvcc 在编排阶段知道 **tileiras 可执行文件** 与 **tile 专用编译目标**
（``TilebcTarget``），与 ``--tile-only`` 模式相关。完整 ``nvcc --help`` 在
13.1 上对 tile 选项的文档仍较少，需结合 verbose 日志与二进制字符串交叉解读。

入口 3：nvlink 链接期 JIT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`05_nvlink` 已列出 elfLink JIT 相关诊断字符串。与本节衔接的要点：

::

   FNLZR: Starting JIT
   FNLZR: Ending JIT
   elfLink JIT compile failed
   elfLink JIT link failed
   Can't JIT TileIR without libtileiras
   Can't JIT TileIR with this driver or tool

当 fatbin / cubin 中含有 **尚未最终化为目标 SM SASS 的 TileIR image** 时，
nvlink 在链接阶段 ``dlopen("libnvidia-tileiras.so.*")`` 并调用
``nvTileIRCompileProgram`` 完成 JIT。若驱动包未安装或版本不匹配，则报
上述错误。

入口 4：fatbinary 打包
~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`08_fatbinary` 中 ``nvFatbinAddTileIR`` 与 CLI 选项：

::

   --tileir-cmdline <options>     (-tileir-cmdline)
   The TileIR command-line options with which the device code is compiled

TileIR image 与 PTX、ELF cubin、LTO IR 一样，作为 fatbin 中的一种
``kind`` 写入容器；``cuobjdump`` 可列出（见下节）。

--------------

cuobjdump：检查 fatbin 中的 TileIR
-------------------------------------

CUDA 13.1 的 ``cuobjdump`` 增加 TileIR 专用开关（``strings cuobjdump``）：

::

   -ltileir / --list-tileir     列出 fatbin 内所有 TileIR 文件
   -xtileir / --extract-tileir  按名称提取 TileIR 字节码
   -dump-tileir                转储 TileIR（字符串提示仍 unimplemented）

对 ``examples/build/vector_add``：

::

   $ cuobjdump -ltileir vector_add
   (无 TileIR 条目 — 符合预期)

若使用 cuTile 编译含 tile kernel 的程序，应在此看到 TileIR 文件名列表，
再用 ``-xtileir`` 导出供 ``tileiras`` 离线重放。

--------------

与 vector_add 的关系
-----------------------

``vector_add`` 使用 ``-arch=sm_89`` 的经典 SIMT pipeline：

::

   cicc → vector_add.ptx → ptxas → vector_add.sm_89.cubin
   fatbinary --image3=kind=ptx,... --image3=kind=elf,...

全程 **无** ``tileiras`` execve、无 ``nvFatbinAddTileIR``、``cuobjdump -ltileir``
为空。TileIR 栈面向 **Blackwell（sm_100+）** 及 cuTile / ``__tile__`` 代码，
与 sm_89 向量加法样例 **正交**——这也是两套 IR 并存的证据。

--------------

Python 生态与版本对齐
------------------------

``pip install cuda-tile`` 可安装 cuTile 编译器包；可选依赖
``nvidia-cuda-tileiras`` 会在 venv 内提供 **另一份** tileiras：

::

   .../site-packages/nvidia/cu13/bin/tileiras
   Cuda compilation tools, release 13.3, V13.3.36

与系统 CTK ``/usr/local/cuda/bin/tileiras`` (13.1.80) **版本可不同**。
NVIDIA 文档要求 cuTile / tileiras / nvvm / nvcc 的 **主次版本一致**；
混用 13.1 系统 CTK 与 13.3 pip tileiras 可能导致 ``nvTileIRIsCompatible``
失败。

--------------

与其他章节的边界
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 32

   * - 主题
     - 章节
   * - nvlink elfLink / FNLZR JIT 概览
     - :doc:`05_nvlink`
   * - fatbin 多 kind 打包
     - :doc:`08_fatbinary`
   * - SIMT cicc / ptxas 主路径
     - :doc:`03_cicc`、:doc:`04_ptxas`
   * - TileIR 入口、tileiras、nvTileIR API
     - 本节

--------------

*分析基于 CUDA 13.1 (CTK build 37061995) / tileiras V13.1.80 / Driver 595.58.03。*

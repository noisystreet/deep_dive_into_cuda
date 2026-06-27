Fat Binary 结构分析
===================

   分析 CUDA fat binary 的容器格式、ELF cubin
   结构以及最终可执行文件中的布局

   环境: CUDA 13.1 / sm_89 (Ada Lovelace)

概述
----

Fat Binary 是 CUDA 实现 **GPU 架构兼容性** 的核心机制。一个 fat binary
可以同时包含针对多种 GPU 架构编译的二进制代码和 PTX 回退代码，运行时
CUDA 驱动自动选择最合适的版本。

完整的 fat binary 数据流包含三个层级：

::

   vector_add.fatbin (容器)   →   嵌入到   →   vector_add.fatbin.c (C数组)
                                                       ↓
        链接到 vector_add 可执行文件的 .nv_fatbin 段中
                                                       ↓
     运行时通过 .nvFatBinSegment 段中的 Wrapper 结构体找到 fatbin

--------------

Fat Binary 容器格式
----------------------

``vector_add.fatbin`` 是一个 4KB 的
**自定义容器格式**\ ，它不是一个标准的
ELF，而是一个带自描述头的容器文件。

容器头部
~~~~~~~~~~~~

::

   偏移   十六进制                                ASCII
   0000:  50 ed 55 ba 01 00 10 00  d0 0f 00 00 00 00 00 00   P.U.............
   0010:  01 00 01 01 50 00 00 00  98 01 00 00 00 00 00 00   ....P...........
   0020:  96 01 00 00 40 00 00 00  01 00 09 00 59 00 00 00   ....@.......Y...
   0030:  00 00 00 00 00 00 00 00  11 80 00 00 00 00 00 00   ................
   0040:  00 00 00 00 00 00 00 00  0e 04 00 00 00 00 00 00   ................
   0050:  48 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00   H...............

==== ==== ========================= ==============================
偏移 大小 值                        说明
==== ==== ========================= ==============================
0x00 8 B  ``50ed55ba 01001000``     **Magic**: fat binary 魔数
0x08 8 B  ``0x00000fd0`` **(4048)** 总长度 = 4048 字节
0x10 4 B  ``0x00010001``            版本号 (v1)
0x14 4 B  ``0x00000050``            头部偏移 = 80 字节
0x18 4 B  ``0x00000198``            头部大小 = 408 字节
0x1c 4 B  ``0x00000040``            **Image 计数 = 2** (PTX + ELF)
0x20 …    …                         Image 目录项
==== ==== ========================= ==============================

Image 目录
~~~~~~~~~~~~~~

容器在头部之后包含一个 **image 目录**\ ，每个条目描述一个已嵌入的
image：

+-----------+-----------+-----------+-----------+-----------+-----------+
| 条目      | 偏移      | 类型      | 架构      | 偏移量    | 大小      |
+===========+===========+===========+===========+===========+===========+
| **Image   | 0x28      | ``0x0     | sm_89     | 0         | 0         |
| 1**       |           | 0000001`` |           | x00000000 | x0000040e |
|           |           | (PTX)     |           | (相对     | (1038 B)  |
|           |           |           |           | 于数据区) |           |
+-----------+-----------+-----------+-----------+-----------+-----------+
| **Image   | 0x40      | ``0x0     | sm_89     | 0         | 0         |
| 2**       |           | 0000002`` |           | x0000045e | x00000da8 |
|           |           | (ELF)     |           |           | (3496 B)  |
+-----------+-----------+-----------+-----------+-----------+-----------+

从 ``vector_add.fatbin.c`` 的汇编编码序列化头部中，首次出现的两个 quad
解释如下：

.. code:: asm

   .section .nv_fatbin, "a"
   .align 8
   fatbinData:
   // 头部: magic + 总大小 + 版本 + 头部
   .quad 0x00100001ba55ed50, 0x0000000000000fd0, 0x0000005001010001, 0x0000000000000198
   // Image 1 条目 (PTX)
   .quad 0x0000004000000196, 0x0000005900090001
   // Image 2 条目 (ELF/SASS)
   .quad 0x0000000000000000, 0x0000000000008011
   // ...

容器内的数据
~~~~~~~~~~~~~~~~

容器内按顺序依次是 PTX 文本和 ELF
cubin，两者都是\ **明文未压缩**\ 存储：

::

   偏移范围          内容
   ──────────────────────────────────────────────────
   0x000000 - 0x00003E   Fat binary 容器头部  (62 B)
   0x00003E - 0x00045D   Image 1: PTX 文本    (1038 B)  ← vector_add.ptx 内容
   0x00045E - 0x000FFD   Image 2: ELF cubin   (3496 B)  ← vector_add.sm_89.cubin 内容

PTX 文本以明文形式嵌入，以 ``"//\n"`` 开头，ELF cubin 则直接内嵌 ELF
文件格式。

--------------

ELF Cubin 结构 (sm_89)
-------------------------

``vector_add.sm_89.cubin`` 是一个 **标准的 ELF64 可执行文件**\ ，专为
NVIDIA CUDA 架构设计。

ELF 头
~~~~~~~~~~

::

   ELF Header:
     Magic:   7f 45 4c 46  02 01 01 41  08 00 00 00  00 00 00 00   ← \x7fELF + 64位 + LE + ABI=0x41
     Type:    EXEC (ET_EXEC)                                          ← 可执行镜像
     Machine: NVIDIA CUDA architecture                                 ← 目标为 GPU
     Flags:   0x6005904                                                ← sm_89 编码
     Entry:   0x0                                                     ← 无 CPU 入口点

**关键特点**\ ： - ``OS/ABI = 0x41`` — NVIDIA CUDA ABI 标识 -
``Type = ET_EXEC`` — cubin 是完整的可执行镜像（非 .o 或 .so），可直接由
GPU 加载 - ``Machine`` 被报告为 “NVIDIA CUDA architecture” - Entry point
为 0 — GPU 代码的入口由 ``__cudaRegisterEntry`` 指定

Section 布局
~~~~~~~~~~~~~~~~

.. code:: text

   [Nr] 名称                                 类型      地址     偏移    大小    旗标
   ───  ────────────────────────────────────  ───────  ───────  ──────  ─────  ──
   [ 0] (null)                               NULL
   [ 1] .shstrtab                             STRTAB   00000000  000040  0015f  (section 名称表)
   [ 2] .strtab                               STRTAB   00000000  00019f  00177  (符号字符串表)
   [ 3] .symtab                               SYMTAB   00000000  000318  000d8  (符号表)
   [ 4] .debug_frame                          PROGBITS 00000000  0003f0  00070  (调试展开信息)
   [ 5] .note.nv.tkinfo                       NOTE     00000000  000460  000a4  (kernel 信息)
   [ 6] .note.nv.cuinfo                       NOTE     00000000  000504  00020  (编译信息)
   [ 7] .nv.info                              (LOPROC) 00000000  000524  00024  (全局 CUDA 元数据)
   [ 8] .nv.info._Z10vector_addPKfS0_Pfi      (LOPROC) 00000000  000548  0006c  (kernel 元数据)
   [ 9] .nv.callgraph                         (LOPROC) 00000000  0005b4  00020  (调用图)
   [10] .nv.rel.action                        (LOPROC) 00000000  0005d8  00010  (重定位信息)
   [11] .rel.debug_frame                      REL      00000000  0005e8  00010  (调试段重定位)
   [12] .nv.constant0._Z10vector_addPKfS0_Pfi PROGBITS 00000000  0005f8  0017c  (常量 bank 0)
   [13] .text._Z10vector_addPKfS0_Pfi         PROGBITS 00000000  000780  00200  (SASS 机器码)

关键 Section 详解
~~~~~~~~~~~~~~~~~~~~~

``.text._Z10vector_addPKfS0_Pfi`` — SASS 机器码 (512 B)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**最重要的段**\ ，包含了 kernel 的实际 GPU 指令。大小 0x200 = 512 字节。

::

   START:  /*0070*/   MOV R1, c[0x0][0x168] ;       ← 设置栈指针
           /*0080*/   S2R R0, SR_CTAID.X ;            ← R0 = blockIdx.x
           /*0090*/   S2R R5, SR_TID.X ;              ← R5 = threadIdx.x
           /*00a0*/   S2R R7, SR_TID.X ;              ← R7 = threadIdx.x  
           /*00b0*/   IMAD R6, R0, c[0x0][0x178], R5 ;← R6 = idx = blockIdx.x * blockDim.x + threadIdx.x
           /*00c0*/   ISETP.GE.AND P0, PT, R6, c[0x0][0x170], PT ; ← idx >= n ?
           /*00d0*/   NOP;                             ← 预测执行（条件跳转前的延迟槽）
           /*00e0*/  @P0 BRA 0x100;                    ← 如果 idx >= n 跳转到 EXIT
           /*00f0*/   MOV R2, c[0x0][0x160]            ← R2 = a (基址)
           /*0100*/   MOV R3, c[0x0][0x164]            ← R3 = a (高32位)
           /*0110*/   MOV R4, c[0x0][0x168]            ← R4 = b (基址)
           /*0120*/   MOV R5, c[0x0][0x16c]            ← R5 = b (高32位)
           /*0130*/   IMAD.WIDE R2, R6, 4, R2          ← R2:R3 = &a[idx] (64位)
           /*0140*/   IMAD.WIDE R4, R6, 4, R4          ← R4:R5 = &b[idx] (64位)
           /*0150*/   LDG.E R4, [R4.64]                ← b[idx] → R4 (全局加载)
           /*0160*/   LDG.E R3, [R2.64]                ← a[idx] → R3 (全局加载)
           /*0170*/   IMAD.WIDE R6, R6, 4, c[0x0][0x170] ← R6:R7 = &c[idx]
           /*0180*/   FADD R9, R4, R3                  ← R9 = a[idx] + b[idx]
           /*0190*/   STG.E [R6.64], R9               ← 写回 c[idx] (全局存储)
           /*01a0*/   EXIT                              ← kernel 结束
           /*01b0*/   BRA 0x1b0;                        ← 死循环（安全兜底）
           /* 后续 NOP 填充到 0x200 边界 */

**实际 SASS vs PTX 对比**\ ：

+-----------------------+-----------------------+-----------------------+
| PTX (虚拟)            | SASS (实际硬件)       | 说明                  |
+=======================+=======================+=======================+
| ``mo                  | `                     | 从特殊寄存器读取      |
| v.u32 %r3, %ctaid.x`` | `S2R R0, SR_CTAID.X`` | blockIdx              |
+-----------------------+-----------------------+-----------------------+
| ``mad.lo.s32``        | ``IMAD R6, R0         | 乘加指令，blockDim    |
|                       | , c[0x0][0x178], R5`` | 来自常量 bank         |
+-----------------------+-----------------------+-----------------------+
| `                     | ``IMAD                | 地址计算合并为宽乘加  |
| `cvta.to.global.u64`` | .WIDE R2, R6, 4, R2`` |                       |
+-----------------------+-----------------------+-----------------------+
| ``ld.global.f32``     | ``LDG.E R4, [R4.64]`` | 全局加载（带 .E       |
|                       |                       | 缓存提示）            |
+-----------------------+-----------------------+-----------------------+
| ``st.global.f32``     | ``STG.E [R6.64], R9`` | 全局存储              |
+-----------------------+-----------------------+-----------------------+
| ``ret``               | ``EXIT``              | 返回指令              |
+-----------------------+-----------------------+-----------------------+

**SASS 与 PTX 的核心差异**\ ： 1. **物理寄存器** — SASS 使用 ``R0-R9``
物理寄存器，PTX 使用虚拟寄存器 ``%r<6>``, ``%f<4>``, ``%rd<11>`` 2.
**常量 bank** — ``c[0x0][0x168]`` 表示从 constant bank 0 的偏移 0x168
处读取，存放 kernel 参数和特殊值 3. **指令格式** — SASS 指令是 64 位编码
(如 ``0x00005c0006067625``)，PTX 是文本格式 4. **延迟槽** — SASS 包含了
NOP 填充用于指令调度，PTX 中没有

``.nv.info._Z10vector_addPKfS0_Pfi`` — Kernel 元数据
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

包含 CUDA 运行时需要的 kernel 属性信息：

::

   Attribute:      EIATTR_CUDA_API_VERSION      0x83       ← CUDA API 版本 13.1
   Attribute:      EIATTR_PARAM_CBANK           0x4 0x1c0160 ← 参数在 constant bank 中的布局
   Attribute:      EIATTR_CBANK_PARAM_SIZE      0x1c       ← 参数总大小 = 28 字节
   Attribute:      EIATTR_KPARAM_INFO           Index:0    ← 参数 0 (a): offset=0x18, size=8  (指针)
                                                Index:1    ← 参数 1 (b): offset=0x10, size=8  (指针)
                                                Index:2    ← 参数 2 (c): offset=0x8,  size=8  (指针)
                                                Index:3    ← 参数 3 (n): offset=0x0,  size=4  (int)
   Attribute:      EIATTR_MAXREG_COUNT          0xff       ← 最大寄存器数 255
   Attribute:      EIATTR_MERCY_ISA_VERSION     0.0        ← ISA 版本
   Attribute:      EIATTR_EXIT_INSTR_OFFSETS    0x50 0xf0  ← EXIT 指令偏移

**参数映射关系**\ ：

::

   kernel<<<(d_a, d_b, d_c, n)>>>
                               ⬇
   constant bank 0x0:
     c[0x0][0x18] = n        (int,    4 B)  ← 参数 3
     c[0x0][0x10] = c        (ptr,    8 B)  ← 参数 2
     c[0x0][0x08] = b        (ptr,    8 B)  ← 参数 1
     c[0x0][0x00] = a        (ptr,    8 B)  ← 参数 0

``.note.nv.tkinfo`` — Kernel 信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

包含 kernel 的调试和性能分析信息，CUDA 工具 (nvprof/nsys) 使用。

``.note.nv.cuinfo`` — 编译信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

包含编译工具版本、编译选项等元数据。

符号表
~~~~~~~~~~

::

   符号表 '.symtab' 包含 9 个条目：
       值        大小    类型    绑定     Ndx    名称
       0x00000000   512  FUNC    GLOBAL   .text  _Z10vector_addPKfS0_Pfi

只有一个全局符号，即 mangled 后的 kernel 函数名。512 字节对应 SASS
代码段大小。

--------------

最终可执行文件中的布局
-------------------------

关键 Section
~~~~~~~~~~~~~~~~

::

   [Nr] 名称              类型    地址              偏移        大小     旗标
   [17] __nv_module_id    PROGBITS 0x000000000009a220  0009a220  0000a0  
   [18] .nv_fatbin        PROGBITS 0x000000000009a230  0009a230  000fe0  ← fat binary 数据
   [31] .nvFatBinSegment  PROGBITS 0x00000000000b5558  000b5558  00018   ← fat binary wrapper

从 ``.fatbin.c`` 到可执行文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vector_add.fatbin.c`` 通过内联汇编在 ``.nv_fatbin`` 段中创建 fat
binary 数据块：

.. code:: c

   // fatbin.c 中的汇编块
   asm(
   ".section .nv_fatbin, \"a\"\n"          // 创建 .nv_fatbin 段
   ".align 8\n"
   "fatbinData:\n"                          // 标签
   ".quad 0x00100001ba55ed50, ..."          // fat binary 内容（496 个 quad）
   );

   // C 结构体：运行时使用的 fat binary wrapper
   static const __fatBinC_Wrapper_t __fatDeviceText
       __attribute__((aligned(8)))
       __attribute__((section(".nvFatBinSegment"))) =  // 放入独立段
       { 0x466243b1, 1, fatbinData, 0 };    // magic + 版本 + 指针 + 标志

编译后，这两个段在可执行文件中的布局：

::

   地址空间                             内容
   ────────────────────────────────────────────────────────────────────
   0x0009a230   .nv_fatbin (0xfe0 B)   ← fatbin 容器（整个 .fatbin 文件）
                                          包含 PTX + ELF cubin 的打包数据
                                          
   0x000b5558   .nvFatBinSegment        ← 16 字节 Wrapper 结构体
                0x466243b1 (magic)          FatBin 段魔数
                0x00000001 (version)        版本
                0x0009a230 (data ptr)       指向 .nv_fatbin 段
                0x00000000 (flags)          标志

运行时的加载流程
~~~~~~~~~~~~~~~~~~~~

::

   程序启动 (main 之前)
           │
           ├─ __sti____cudaRegisterAll()      ← stub.c 中的 constructor
           │      │
           │      └─ __cudaRegisterBinary()   ← 遍历 .nvFatBinSegment
           │             │
           │             └─ 找到 __fatDeviceText
           │                    │
           │                    解析 fatbin 容器头
           │                    │
           │                    ├─ Image 1: PTX (后备/JIT)
           │                    └─ Image 2: ELF cubin
           │                           │
           │                           匹配当前 GPU 架构?
           │                           ├─ 匹配 → 直接加载 ELF 到 GPU
           │                           └─ 不匹配 → JIT 编译 PTX → 加载
           │
           ├─ __cudaRegisterEntry()          ← 注册 kernel 符号
           │      │
           │      └─ 将 _Z10vector_addPKfS0_Pfi 与 GPU 函数入口关联
           │
           └─ main() 执行到 vector_add<<<>>>
                  │
                  └─ __cudaLaunch()          ← 通过已注册的入口启动 kernel

--------------

多层封装关系总结
-------------------

::

   ┌──────────────────────────────────────────────────────────────────────┐
   │                      vector_add (可执行文件)                          │
   │                                                                      │
   │   ┌─────────────────────────────────────────────────────────────┐    │
   │   │  .nvFatBinSegment                                           │    │
   │   │   __fatBinC_Wrapper_t { magic, version, ptr → fatbin, 0 }  │    │
   │   └─────────────────────────────────────────────────────────────┘    │
   │                             │ 指向                                    │
   │                             ▼                                         │
   │   ┌─────────────────────────────────────────────────────────────┐    │
   │   │  .nv_fatbin (fat binary 容器)                                │    │
   │   │                                                              │    │
   │   │  ┌──────────────────────────────────────────────────┐       │    │
   │   │  │  FB Header: magic + count(2) + offset table      │       │    │
   │   │  ├──────────────────────────────────────────────────┤       │    │
   │   │  │  Image 1: PTX (sm_89)                           │       │    │
   │   │  │    .version 9.1                                  │       │    │
   │   │  │    .target sm_89                                 │       │    │
   │   │  │    ...                                           │       │    │
   │   │  ├──────────────────────────────────────────────────┤       │    │
   │   │  │  Image 2: ELF cubin (sm_89)                     │       │    │
   │   │  │    ┌──────────────────────────────────────┐    │       │    │
   │   │  │    │  ELF Header (ET_EXEC, ABI=0x41)      │    │       │    │
   │   │  │    ├──────────────────────────────────────┤    │       │    │
   │   │  │    │  .text._Z10vector_addPKfS0_Pfi       │    │       │    │
   │   │  │    │    SASS 机器码 (0x200 B / 512 B)    │    │       │    │
   │   │  │    ├──────────────────────────────────────┤    │       │    │
   │   │  │    │  .nv.info (kernel 元数据)            │    │       │    │
   │   │  │    │  .nv.constant0 (kernel 参数布局)     │    │       │    │
   │   │  │    │  .nv.callgraph / .nv.rel.action      │    │       │    │
   │   │  │    │  .note.nv.tkinfo / .note.nv.cuinfo   │    │       │    │
   │   │  │    │  .debug_frame                        │    │       │    │
   │   │  │    │  .symtab / .strtab / .shstrtab       │    │       │    │
   │   │  │    └──────────────────────────────────────┘    │       │    │
   │   │  └──────────────────────────────────────────────────┘       │    │
   │   └─────────────────────────────────────────────────────────────┘    │
   └──────────────────────────────────────────────────────────────────────┘

--------------

cubin vs fatbin 对比
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 维度
     - ``.cubin`` (CUDA Binary)
     - ``.fatbin`` (Fat Binary)
   * - 本质
     - 标准 ELF64 可执行文件
       (``ET_EXEC``)
     - 自定义容器格式（自描述头 + image 目录 + 数据）
   * - 内容
     - 单一架构的 SASS 机器码 +
       元数据 (.text / .nv.info /
       .nv.constant)
     - 多个 image 打包：
       PTX 文本 + 多架构 ELF cubin
   * - 文件结构
     - ELF header + program/section
       headers + .text + .symtab 等
     - 62 B 容器头 + image 目录 +
       连续 image 数据块（明文无压缩）
   * - 大小
     - 3.5 KB (vector_add)
     - 4 KB (vector_add) — 多出 PTX +
       容器头部开销
   * - 生成阶段
     - ptxas 输出（PTX → SASS）
     - fatbinary 输出（打包 cubin + PTX）
   * - 链接关系
     - 可被 nvlink 链接（多个 cubin
       合并为一个）
     - 不参与链接——仅作为数据嵌入
       ``.fatbin.c``
   * - 运行时
     - 由驱动直接加载到 GPU 执行
     - 驱动解析容器头，根据 GPU 型号
       选择合适 image 再加载
   * - 架构兼容性
     - 不可跨架构（sm_89 的 cubin
       无法在 sm_90 上运行）
     - 可跨架构（同时打包多架构 cubin
       及 PTX 回退）

关键发现
-----------

1. **容器 vs 内容** — ``fatbin`` 是容器格式，\ ``cubin`` 是 ELF
   格式。容器里可以包多个 cubin。

2. **PTX 和 SASS 同时存在** — 即使针对单一架构 (sm_89)，fat binary
   也同时包含 PTX 和 SASS：

   -  SASS (ELF cubin) = **直接执行**\ ，零编译开销
   -  PTX = **JIT 回退**\ ，当驱动版本与编译版本不兼容时使用

3. **2 次 fatbinary 调用** — 编译过程中 ``fatbinary`` 被调用了 2 次：

   -  第 1 次：打包单个 cubin 的 SASS + PTX → 生成嵌入 ``.o`` 文件
   -  第 2 次（after nvlink）：打包链接后的 device cubin →
      生成最终的设备链接 fat binary

4. **参数通过 constant bank 传递** — kernel
   参数不是通过寄存器传递，而是通过 **constant memory bank 0**
   (``c[0x0][...]``)，所有线程共享，支持缓存。

5. **运行时自动选择** — CUDA 驱动在加载时根据当前 GPU 计算能力选择最优的
   image：

   -  精确匹配 sm_89 → 使用 ELF cubin (SASS)
   -  不匹配 → 用 PTX JIT 编译为当前硬件指令

.. mermaid:: ../_static/fatbinary_hierarchy.mmd

--------------

*所有文件可在 ``build/`` 目录中找到。*

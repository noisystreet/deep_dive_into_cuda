SASS 执行分析
=================

   :doc:`03_fatbinary` 在 ELF 层面定位了 ``.text`` 段中的 SASS 机器码；
   :doc:`../chapter_02_toolchain/04_ptxas` 从编译器视角说明 ptxas 如何生成
   这些指令。本节换到 **硬件执行视角**：当 ``vector_add<<<4096, 256>>>`` 被
   调度到 sm_89 的 SM 上时，一个 warp 里的 32 条线程如何 **同步取指、谓词
   分歧、合并访存**。

   分析对象：``examples/vector_add.cu`` 在 CUDA 13.1 / sm_89 下生成的
   ``vector_add.cubin``。

   环境: CUDA 13.1 / sm_89 (Ada Lovelace) / Linux x86-64

.. admonition:: 你知道吗？

   GPU 没有「每个线程一个 PC」的传统多线程模型。一个 warp 的 32 个线程
   **共享同一条指令流**，靠 **谓词寄存器** 决定哪些 lane 真正写回结果。
   因此 ``if (idx < n)`` 在 SASS 里不会变成每个线程独立跳转，而是
   ``ISETP`` 设置 ``P0``，再用 ``@P0 EXIT`` 让越界 lane 提前退出——其余
   lane 继续执行 load/add/store。

--------------

与前文的关系
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 32

   * - 章节
     - 视角
   * - :doc:`03_fatbinary`
     - cubin ELF 段布局、SASS 与 PTX 对照表
   * - :doc:`../chapter_02_toolchain/04_ptxas`
     - ptxas 两阶段流水线、指令编码字段
   * - :doc:`../chapter_03_runtime/03_kernel_launch`
     - launch 如何把 grid/block/参数写入 constant bank
   * - **本节**
     - warp 级执行语义、资源占用与访存合并

--------------

复现与分析工具
------------------

.. code:: bash

   nvcc --keep --generate-code=arch=compute_89,code=sm_89 -O2 \
        -Xptxas=-v examples/vector_add.cu -o /tmp/vector_add
   cuobjdump -sass vector_add.cubin
   cuobjdump -res-usage vector_add.cubin
   nvdisasm vector_add.cubin | head -80

.. list-table::
   :header-rows: 1
   :widths: 22 38

   * - 工具
     - 本节用途
   * - ``cuobjdump -sass``
     - 带 PC 偏移的 SASS 汇编，对照 PTX 语义
   * - ``cuobjdump -res-usage``
     - 每 kernel 寄存器 / constant / spill 汇总
   * - ``ptxas -v``
     - 编译期资源统计，与 cubin 内 ``.nv.info`` 互证
   * - ``nvdisasm``
     - 原始 hex 与段边界，验证 64-bit 指令对齐

--------------

资源 footprint：为何能跑满 occupancy
--------------------------------------

``ptxas -v`` 与 ``cuobjdump -res-usage`` 对 ``_Z10vector_addPKfS0_Pfi`` 给出一致结论：

::

   ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
   cuobjdump     : REG:12 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:380

.. list-table::
   :header-rows: 1
   :widths: 22 18 40

   * - 资源
     - 用量
     - 执行含义
   * - 通用寄存器
     - 12 / 255
     - 仅 ``R0–R9`` 与少量 UR 参与热路径；无 spill
   * - constant bank 0
     - 380 B
     - launch 写入 blockDim、n、指针 a/b/c 等；kernel 通过 ``c[0x0][off]`` 读取
   * - shared / local
     - 0
     - 无 ``__shared__``、无栈帧 spill
   * - ``.text`` 段
     - 512 B
     - 17 条有效指令 + NOP 填充至 0x200 边界

在 sm_89 上，65536 个 32-bit 寄存器 / SM 可支持远多于 48 warp 上限，因此
**12 寄存器不会成为 occupancy 瓶颈**——限制来自硬件每 SM 最多 48 warp
（1536 线程）。对 ``<<<4096, 256>>>`` 这种大 grid，scheduler 可持续派发新 block。

--------------

执行模型：一个 warp 如何跑完 kernel
--------------------------------------

.. mermaid:: ../_static/sass_execution.mmd

SIMT 三原则（结合本 kernel 验证）：

1. **同 warp 同 PC** — 32 个 lane 从 ``/*0000*/`` 同步推进，除非遇到分歧。
2. **谓词屏蔽写回** — ``@P0 EXIT`` 仅对 ``P0=1`` 的 lane 生效；``P0=0`` 的
   lane 把该指令当 NOP。
3. **内存合并** — 当相邻 lane 的 ``idx`` 连续且地址对齐时，``LDG.E`` /
   ``STG.E`` 由 L1/TLS 合并为宽事务。

对本 launch（``n = 1\,048\,576``，``blocks = 4096``，``threads = 256``）：

-  总线程数 ``4096 × 256 = n``，**每个 lane 的 idx 都有效**，``P0`` 恒为 0，
   运行时 **不会出现谓词分歧**。
-  ``@P0 EXIT`` 仍被保留，因为 ``n`` 是运行时参数；若 ``n`` 不是 block 大小的
   整数倍，最后一个 block 才会产生 tail 分歧。
-  每个 warp 内 ``idx, idx+1, …, idx+31`` 连续 → 对 ``float`` 数组的
   ``LDG.E`` / ``STG.E`` **完全合并**。

--------------

指令级走读：从 prologue 到 store
----------------------------------

完整 ``cuobjdump -sass`` 输出见 cubin；下面按 **执行顺序** 解读关键指令
（PC 为 cubin 内偏移）。

Prologue：线程坐标与边界谓词
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

   /*0010*/  S2R R6, SR_CTAID.X ;           blockIdx.x → R6
   /*0020*/  S2R R3, SR_TID.X ;             threadIdx.x → R3
   /*0030*/  IMAD R6, R6, c[0x0][0x0], R3 ; idx = blockIdx * blockDim + threadIdx
   /*0040*/  ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT ; P0 = (idx >= n)
   /*0050*/  @P0 EXIT ;                     越界 lane 直接退出

对照 PTX：

.. code:: text

   mad.lo.s32  %r1, %r3, %r4, %r5;    // idx
   setp.ge.s32 %p1, %r1, %r2;         // idx >= n
   @%p1 bra    $L__BB0_2;             // 跳转到 ret

ptxas 将 PTX 的条件 **反转** 为 ``idx >= n`` 并映射到 ``@P0 EXIT``，语义等价于
``if (idx >= n) return``。``c[0x0][0x0]`` 存放 launch 时的 ``blockDim.x``，
``c[0x0][0x178]`` 存放 ``n``——偏移由 launch stub 按 ABI 写入，不同 ptxas
版本可能调整具体 offset，但 **「参数在 constant bank、S2R 读特殊寄存器」**
的模式不变。

热路径：地址计算与 load/add/store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

   /*0060*/  MOV R7, 0x4 ;
   /*0070*/  ULDC.64 UR4, c[0x0][0x118] ;
   /*0080*/  IMAD.WIDE R4, R6, R7, c[0x0][0x168] ;  &b[idx]
   /*0090*/  IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160] ;  &a[idx]
   /*00a0*/  LDG.E R4, [R4.64] ;                        b[idx]
   /*00b0*/  LDG.E R3, [R2.64] ;                        a[idx]
   /*00c0*/  IMAD.WIDE R6, R6, R7, c[0x0][0x170] ;      &c[idx]
   /*00d0*/  FADD R9, R4, R3 ;
   /*00e0*/  STG.E [R6.64], R9 ;

要点：

-  **IMAD.WIDE** 将 32-bit ``idx`` 乘 4 并与 64-bit 基址相加，对应 PTX 的
   ``mul.wide`` + ``add.s64`` + ``cvta.to.global`` 折叠。
-  **``.reuse``** 提示调度器 ``R6`` / ``R7`` 可在多条指令间复用读端口，属于
   ptxas 调度 hint，不改变语义。
-  **LDG.E** 的 ``.E`` 为 evict-normal 缓存策略；对只读一遍的输入数组合理。
-  **FADD** 为 IEEE 单精度加法，与 PTX ``add.f32`` 一一对应。
-  **STG.E** 写回全局内存；无 ``.cs`` 流式标记，结果驻留 L2 供后续读取。

Epilogue 与兜底块
~~~~~~~~~~~~~~~~~~~~

.. code:: text

   /*00f0*/  EXIT ;
   /*0100*/  BRA 0x100;    // 自旋 — 不应被正常路径到达
   /*0110*/–/*01f0*/  NOP ;  // 填充至 512 B 边界

正常 lane 在 ``/*00f0*/`` 以 ``EXIT`` 结束。``/*0100*/`` 起的 ``BRA 0x100``
是 ptxas 插入的 **trap 兜底**——若控制流异常落入此地址，warp 会自旋而非
执行随机内存。NOP 序列用于 **指令 cache line 对齐** 与调度占位，不参与逻辑。

--------------

谓词分歧：tail block 里发生了什么
------------------------------------

设 ``n = 1\,048\,576``，``blockDim.x = 256``，``gridDim.x = 4096``。

.. list-table::
   :header-rows: 1
   :widths: 18 22 40

   * - 配置
     - idx 范围
     - warp 行为
   * - ``n = 1\,048\,576``，4096×256 launch
     - 每个 lane 满足 ``idx < n``
     - 无 ``@P0 EXIT`` 命中，边界检查为「死代码」
   * - ``n = 1\,048\,575``，同上 grid
     - 末 block 最后一个 lane 越界
     - 1 个 lane ``@P0 EXIT``，同 warp 其余 31 lane 继续 — **典型分歧**
   * - 任意 ``n < grid 总线程数``
     - 末 block 部分 lane 越界
     - 分歧仅出现在 grid 尾部

--------------

constant bank 与 launch 参数的衔接
------------------------------------

:doc:`../chapter_03_runtime/03_kernel_launch` 说明参数经 driver 写入 constant
memory。SASS 侧不再 ``ld.param``——launch 完成后直接读 ``c[0x0][offset]``：

.. list-table::
   :header-rows: 1
   :widths: 20 28 32

   * - SASS 引用
     - 语义
     - 对应源码参数
   * - ``c[0x0][0x0]``
     - blockDim.x
     - ``<<<blocks, 256>>>`` 中的 256
   * - ``c[0x0][0x178]``
     - n
     - ``vector_add(..., n)``
   * - ``c[0x0][0x160]``
     - 指针 a 的 64-bit 地址
     - ``d_a``
   * - ``c[0x0][0x168]``
     - 指针 b
     - ``d_b``
   * - ``c[0x0][0x170]``
     - 指针 c
     - ``d_c``

``cuobjdump -res-usage`` 报告的 ``CONSTANT[0]:380`` 即整块 parameter buffer
大小。``.nv.info`` 段中的 ``EIATTR_KPARAM_INFO`` 记录各参数 offset，与上表一致
（可用 ``cuobjdump -elf`` 查看）。

--------------

与 PTX 的结构性差异（执行相关）
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 34

   * - 维度
     - PTX
     - SASS 执行
   * - 寄存器
     - 无限虚拟寄存器 ``%r/%f/%rd``
     - 12 个物理 ``R*``，由 ptxas 分配
   * - 参数
     - ``ld.param`` 在入口显式加载
     - launch 预填 constant bank，kernel 直接 ``c[0x0][off]``
   * - 分支
     - ``@p bra label``
     - ``ISETP`` + ``@P0 EXIT`` / ``BRA``，warp 级分歧
   * - 全局访问
     - ``ld.global.f32``
     - ``LDG.E`` / ``STG.E``，带缓存与合并硬件行为
   * - 返回
     - ``ret``
     - ``EXIT``，释放 warp slot

更完整的对照表见 :doc:`03_fatbinary`；编码 bit 布局见
:doc:`../chapter_02_toolchain/04_ptxas`。

--------------

延伸：从 vector_add 到 Tensor Core
------------------------------------

:doc:`01_compilation_pipeline` 的 WMMA 对比节说明：同一编译链上，ptxas 可将
``wmma.mma.sync`` 映射为 **HMMA.16816.F32** 而非 ``FADD``。执行层面的差异是：

-  vector_add：每 lane 独立 ``LDG`` + ``FADD`` + ``STG``，受内存带宽限制。
-  wmma_matmul：warp 协作一条 ``HMMA`` 完成 16×16×16 矩阵块，受 Tensor Core
   算力限制，寄存器用量跃升至 100+，occupancy 策略完全不同。

本节分析的 **谓词 / 合并访存 / constant bank** 机制同样适用于 HMMA kernel，
只是热路径指令替换为 Tensor Core 原生指令。

--------------

*分析基于 CUDA 13.1 (build 37061995)，样例 ``examples/vector_add.cu`` 与
``cuobjdump`` / ``nvdisasm`` 输出互证。*

多架构 Fat Binary
=================

   在单架构分析（:doc:`03_fatbinary`）基础上，通过 ``-gencode`` 编译
   ``sm_75 + sm_89`` 双架构 fat binary，对比容器体积、image 组成与 SASS
   差异，并说明 CUDA 驱动如何在运行时选择 image。

   环境: CUDA 13.1 / sm_89 (Ada Lovelace) / Linux x86-64

.. admonition:: 你知道吗？

   为什么需要多架构 fat binary？一个直接的场景是游戏行业：Steam 上
   的 CUDA 应用必须同时支持 GTX 1060 (sm_61) 到 RTX 4090 (sm_89)
   的几十种 GPU。如果只提供 sm_89 的 SASS，Kepler 用户无法运行；
   如果只提供 PTX，驱动 JIT 编译会增加首次启动延迟。NVCC 的
   ``-gencode`` 机制让你可以在一个二进制中同时打包多个 SASS 镜像和
   一个 PTX 回退方案，兼顾性能和兼容性。

--------------

为什么需要多架构 Fat Binary？
------------------------------

:doc:`03_fatbinary` 分析了 ``-arch=sm_89`` 单架构场景：一个 fat binary 包含
**PTX + SASS** 两个 image。这解决了\ **驱动版本兼容** 问题（PTX JIT 回
退），但\ **不能** 让同一份程序在不同代际 GPU 上高效运行——sm_75 设备无法
直接执行 sm_89 的 SASS 指令。

多架构编译通过多次 ``-gencode`` 为每种目标 SM 生成独立 cubin，``fatbinary``
工具将它们打包进同一容器。运行时驱动按当前 GPU 的计算能力（Compute
Capability）选择\ **精确匹配** 的 SASS image，仅在无匹配 cubin 时才回退到
PTX JIT。

本节要回答：

1. 每增加一个 ``-gencode`` 目标，fat binary 体积增加多少？
2. ``fatbinary --create`` 命令如何描述多个 image？
3. sm_75 与 sm_89 的 SASS 有何差异？
4. 驱动在 sm_89 机器上实际选择哪个 image？

--------------

实验设计：三种编译配置
------------------------

基于 ``examples/vector_add.cu``\ ，使用三种 ``nvcc`` 配置对比：

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - 配置名
     - nvcc 命令
     - 预期 image
   * - single
     - ``nvcc -arch=sm_89``
     - PTX sm_89 + SASS sm_89
   * - dual
     - ``-gencode=arch=compute_75,code=sm_75``\n
       ``-gencode=arch=compute_89,code=sm_89``
     - SASS sm_75 + SASS sm_89（无 PTX）
   * - dual_ptx
     - dual 基础上追加\n
       ``-gencode=arch=compute_89,code=compute_89``
     - SASS sm_75 + PTX sm_89 + SASS sm_89

复现命令：

.. code:: bash

   nvcc --keep vector_add.cu -o out_single -arch=sm_89

   nvcc --keep vector_add.cu -o out_dual \
     -gencode=arch=compute_75,code=sm_75 \
     -gencode=arch=compute_89,code=sm_89

   nvcc --keep vector_add.cu -o out_dual_ptx \
     -gencode=arch=compute_75,code=sm_75 \
     -gencode=arch=compute_89,code=sm_89 \
     -gencode=arch=compute_89,code=compute_89

   # 分析
   ls -lh vector_add.fatbin vector_add.compute_*.cubin
   cuobjdump -lelf out_dual_ptx
   cuobjdump -ptx out_dual_ptx
   cuobjdump -sass vector_add.compute_75.cubin

--------------

体积对比：每多一个架构增加多少？
----------------------------------

实测结果（``vector_add``\ ，CUDA 13.1）：

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 14 36

   * - 配置
     - fatbin
     - 可执行文件
     - 增量
     - 模块内 image（fatbinary 第 1 次打包）
   * - single
     - 4,064 B
     - 1,052,416 B
     - —
     - PTX + ELF sm_89
   * - dual
     - 6,880 B
     - 1,056,528 B
     - +2,816 B
     - ELF sm_75 + ELF sm_89
   * - dual_ptx
     - 7,368 B
     - 1,056,528 B
     - +488 B（相对 dual）
     - ELF sm_75 + PTX + ELF sm_89

关键观察：

1. 增加 sm_75 SASS 使 fatbin 增大 2,816 B（≈ ``compute_75.cubin`` 3.2 KB
   加上容器目录开销）。
2. 追加 PTX 回退仅增加 488 B（相对 dual）——PTX 文本 1.3 KB 与已有 sm_89
   cubin 部分重叠复用。
3. 可执行文件增量（+4,112 B）大于 fatbin 增量，因为 device link 阶段会生成
   第二份多架构 fat binary（见下文）。

中间产物大小：

.. code:: text

   vector_add.compute_75.cubin    3,256 B    ← sm_75 SASS
   vector_add.compute_89.cubin    3,584 B    ← sm_89 SASS
   vector_add.ptx                 1,338 B    ← PTX 虚拟 ISA

--------------

fatbinary 如何打包多个 image？
-------------------------------

``nvcc --verbose`` 输出揭示了 ``fatbinary`` 工具的命令行。三种配置的核心
差异：

single（``-arch=sm_89``）：

.. code:: text

   fatbinary --create="vector_add.fatbin" -64 \
     "--image3=kind=elf,sm=89,file=vector_add.sm_89.cubin" \
     "--image3=kind=ptx,sm=89,file=vector_add.ptx" \
     --embedded-fatbin="vector_add.fatbin.c"

dual（两个 SASS，无 PTX）：

.. code:: text

   fatbinary --create="vector_add.fatbin" -64 \
     "--image3=kind=elf,sm=75,file=vector_add.compute_75.cubin" \
     "--image3=kind=elf,sm=89,file=vector_add.compute_89.cubin" \
     --embedded-fatbin="vector_add.fatbin.c"

dual_ptx（SASS × 2 + PTX 回退）：

.. code:: text

   fatbinary --create="vector_add.fatbin" -64 \
     "--image3=kind=elf,sm=75,file=vector_add.compute_75.cubin" \
     "--image3=kind=ptx,sm=89,file=vector_add.compute_89.ptx" \
     "--image3=kind=elf,sm=89,file=vector_add.compute_89.sm_89.cubin" \
     --embedded-fatbin="vector_add.fatbin.c"

``--image3`` 参数格式：

.. code:: text

   --image3=kind=<ptx|elf>,sm=<NN>,file=<path>

每个 ``-gencode=arch=compute_XX,code=sm_XX`` 生成一个 ``kind=elf`` 的
image；``code=compute_XX`` 则生成 ``kind=ptx`` 的 image。``fatbinary`` 按
命令行顺序将它们写入容器的数据区，并在头部维护 image 目录。

.. mermaid:: ../_static/multiarch_fatbinary.mmd

--------------

cuobjdump 验证：可执行文件中的 image
-------------------------------------

``cuobjdump -lelf`` 列出 fat binary 中嵌入的所有 ELF cubin：

dual / dual_ptx 可执行文件（4 个 ELF）：

.. code:: text

   ELF file    1: out.1.sm_75.cubin
   ELF file    2: out.2.sm_89.cubin
   ELF file    3: out.3.sm_75.cubin
   ELF file    4: out.4.sm_89.cubin

为何是 **4 个** 而非 2 个？编译过程有 **两次** ``fatbinary`` 调用（:doc:
`03_fatbinary` 关键发现 #3）：

1. **模块 fatbin** — 打包单个编译单元的 device 代码
2. **device link fatbin** — ``nvlink`` 链接后再次打包

每个 fatbin 都包含 sm_75 + sm_89 两份 cubin，因此 ``cuobjdump`` 报告 4 个
ELF。运行时加载的是 **device link 后的最终 fatbin**。

``cuobjdump -ptx`` 在 dual_ptx 配置下：

.. code:: text

   Fatbin ptx code:
   ================
   arch = sm_89
   code version = [9,1]
   ...
   .version 9.1
   .target sm_89

dual 配置（无 ``code=compute_89``）则 **不包含 PTX image**——``cuobjdump -ptx``
无输出。

--------------

SASS 差异：同一 kernel，不同架构
---------------------------------

``cuobjdump -sass`` 对比 sm_75 与 sm_89 的 cubin，同一 kernel
``_Z10vector_addPKfS0_Pfi`` 的逻辑相同，但 **指令选择与编码不同**：

**sm_75（Turing）** — 使用 ``LDG.E.SYS`` 直接加载：

.. code:: text

   /*0090*/    LDG.E.SYS R4, [R4] ;

**sm_89（Ada）** — 先 ``ULDC.64`` 加载常量地址，再 ``IMAD.WIDE`` 带 ``.reuse``：

.. code:: text

   /*0070*/    ULDC.64 UR4, c[0x0][0x118] ;
   /*0080*/    IMAD.WIDE R4, R6, R7, c[0x0][0x168] ;
   /*0090*/    IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160] ;

这说明 **不能** 在 sm_75 GPU 上运行 sm_89 cubin——必须各自编译。多架构 fat
binary 的价值正在于此：同一份可执行文件携带多套 SASS，驱动按硬件选择。

--------------

运行时 image 选择策略
-----------------------

CUDA 驱动在 ``__cudaRegisterBinary`` 加载 fat binary 时执行 image 选择。
简化决策流程：

.. code:: text

   1. 读取当前 GPU 计算能力 (本机: sm_89)
   2. 在 fat binary 中查找 kind=elf 且 sm 精确匹配的 image
      → 找到 sm_89 cubin → 直接加载 SASS（零 JIT 开销）
   3. 若无精确 cubin 匹配，查找 kind=ptx 的 image
      → 驱动 JIT 编译 PTX 为当前硬件 SASS
   4. 若均无匹配 → 启动失败

在本机 sm_89 上验证：三种配置的可执行文件均输出 ``Max error: 0.000000``\ ，
说明 sm_89 cubin 被正确选中并执行。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 运行环境
     - dual 配置的选择结果
   * - sm_89 (Ada, 本机)
     - ``compute_89.cubin`` SASS — 直接执行
   * - sm_75 (Turing)
     - ``compute_75.cubin`` SASS — 直接执行
   * - sm_90 (Hopper, 无 cubin)
     - dual：**无 PTX → 可能失败**\ ；dual_ptx：PTX JIT 回退

**注意**：dual 配置（仅 SASS、无 PTX）无法在新一代 GPU 上通过 JIT
回退。发布跨架构二进制时，建议至少为\ **最高目标架构** 同时嵌入 PTX
（dual_ptx 模式）。

--------------

-gencode 与 -arch 的关系
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 选项
     - 行为
   * - ``-arch=sm_89``
     - 等价于单条 ``-gencode=arch=compute_89,code=sm_89``\ ，且 NVCC 默认
       同时生成 PTX 回退
   * - ``-gencode=arch=compute_75,code=sm_75``
     - 仅生成 sm_75 SASS cubin，**不** 自动生成 PTX
   * - ``-gencode=arch=compute_89,code=compute_89``
     - 仅生成 PTX（虚拟架构），不含 SASS
   * - 多次 ``-gencode``
     - 每个目标独立调用 cicc + ptxas，最终由 fatbinary 合并

CMake 中对应写法：

.. code:: cmake

   set_target_properties(vector_add PROPERTIES
       CUDA_ARCHITECTURES "75;89"
   )

``CUDA_ARCHITECTURES "75;89"`` 等价于上述 dual 配置——不含 PTX 回退。若需
PTX，需显式追加 ``CUDA_ARCHITECTURES "75;89-virtual"`` 或使用
``CMAKE_CUDA_FLAGS`` 添加 ``-gencode=arch=compute_89,code=compute_89``。

--------------

与单架构 fat binary 的对比总结
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 维度
     - 单架构 (1.3 节)
     - 多架构 (本节)
   * - image 数量
     - 2（PTX + SASS）
     - 2–3+（多个 SASS ± PTX）
   * - fatbin 体积
     - 4 KB
     - 7–8 KB（本实验）
   * - 兼容性
     - 同架构 + PTX JIT
     - 跨架构 SASS + 可选 PTX JIT
   * - 编译时间
     - 1× ptxas
     - N× ptxas（每个 sm_XX 一次）

--------------

关键发现
--------

1. **每个 ``-gencode=code=sm_XX`` 增加一个 ELF image** — 本实验中 sm_75
   cubin 使 fatbin 增大 2,816 B。

2. **PTX 不是自动生成的** — 仅 ``-arch=sm_XX`` 或显式
   ``code=compute_XX`` 才嵌入 PTX；纯 multi-SASS 配置无 JIT 回退能力。

3. **fatbinary 命令行即 image 清单** — ``--image3=kind=...,sm=...,file=...``
   直接对应容器内的 image 目录条目。

4. **同一 kernel 在不同 SM 上 SASS 不同** — sm_75 用 ``LDG.E.SYS``\ ，
   sm_89 用 ``ULDC.64`` + ``IMAD.WIDE.reuse``\ ，不可混用。

5. **device link 产生第二份 fatbin** — ``cuobjdump -lelf`` 报告的 4 个
   cubin = 2 个 SM × 2 次 fatbinary 打包。

6. **驱动优先精确 SASS 匹配** — 本机 sm_89 运行 dual 配置时直接加载
   sm_89 cubin，三种配置结果均正确。

--------------

*Deep Dive Into CUDA — 2026 年 6 月*

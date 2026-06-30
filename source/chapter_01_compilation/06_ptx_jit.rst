PTX JIT 回退
=================

   :doc:`04_multiarch_fatbinary` 说明了 fat binary 里 **何时嵌入 PTX**、驱动
   **优先选 cubin** 的策略。本节聚焦 **JIT 回退本身**：离线编译跳过 ptxas 时
   容器长什么样、``__cudaRegisterFatBinary`` 何时触发进程内 JIT、以及
   ``~/.nv/ComputeCache`` 如何缓存结果。

   分析对象：``examples/vector_add.cu``，对比四种 ``-gencode`` 配置。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) / Linux x86-64

.. admonition:: 你知道吗？

   PTX JIT **不会** ``execve("ptxas", ...)``。与 nvcc 离线链路不同，驱动在
   ``libcuda.so`` 进程内完成 PTX→SASS，但首次 JIT 仍会通过 ioctl 分配 GPU
   侧临时资源，并写入 ``~/.nv/ComputeCache``。第二次启动同一 PTX 时改为
   **只读** 打开缓存文件，跳过重新编译。

--------------

与前文的关系
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 32

   * - 章节
     - 视角
   * - :doc:`03_fatbinary`
     - fatbin 容器内 PTX / ELF 两种 image
   * - :doc:`04_multiarch_fatbinary`
     - 多 ``-gencode`` 与 image 选择策略
   * - :doc:`../chapter_02_toolchain/11_register_chain`
     - ``__cudaRegisterFatBinary`` 加载 fatbin 的时机
   * - :doc:`../chapter_04_runtime_advanced/03_module_loading`
     - ``cuModuleLoadData(PTX)`` 的动态加载 JIT
   * - **本节**
     - 静态 fatbin 的 JIT 触发条件、缓存与离线/在线对比

--------------

实验设计：四种编译配置
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 44 28

   * - 配置
     - nvcc 命令
     - fatbin 内容
   * - ptx_only
     - ``-gencode=arch=compute_89,code=compute_89``
     - **仅 PTX**，无 cubin
   * - single
     - ``-arch=sm_89``
     - PTX + sm_89 cubin
   * - dual_no_ptx
     - ``code=sm_75`` + ``code=sm_89``
     - 两个 cubin，**无 PTX**
   * - dual_ptx
     - dual + ``code=compute_89``
     - 两个 cubin + PTX 回退

复现：

.. code:: bash

   cd /tmp && cp examples/vector_add.cu .
   nvcc -O2 -gencode=arch=compute_89,code=compute_89 vector_add.cu -o vec_ptx_only
   nvcc -O2 -arch=sm_89 vector_add.cu -o vec_single
   nvcc -O2 -gencode=arch=compute_75,code=sm_75 \
        -gencode=arch=compute_89,code=sm_89 vector_add.cu -o vec_dual_no_ptx
   cuobjdump -all vec_ptx_only
   cuobjdump -lelf vec_dual_no_ptx

--------------

离线侧：何时调用 ptxas？
---------------------------

``nvcc --verbose`` 对比 **ptx_only** 与 **single**：

.. code:: text

   # ptx_only — 仅 cicc + fatbinary（无 ptxas）
   cicc ... -arch compute_89 ... -o vector_add.ptx
   fatbinary ... --image3=kind=ptx,sm=89,file=vector_add.ptx ...

   # single — cicc + ptxas + fatbinary（双 image）
   cicc ... -arch compute_89 ... -o vector_add.ptx
   ptxas -arch=sm_89 -m64 vector_add.ptx -o vector_add.sm_89.cubin
   fatbinary ... --image3=kind=elf,sm=89,file=vector_add.sm_89.cubin \
                 --image3=kind=ptx,sm=89,file=vector_add.ptx ...

**结论**：``code=compute_XX`` 只把 PTX 打进 fatbin；``code=sm_XX`` 才在
**编译期** 调用 ptxas。JIT 回退把 ptxas 的工作推迟到 **首次模块加载**。

``cuobjdump -all vec_ptx_only`` 输出仅含：

::

   Fatbin ptx code:
   arch = sm_89
   code version = [9,1]
   compressed

``cuobjdump -lelf vec_ptx_only`` 报 **No ELF file found**——与预期一致。

--------------

体积：JIT 回退省下了什么？
---------------------------

``readelf -SW`` 实测 ``.nv_fatbin`` 段：

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 36

   * - 配置
     - ``.nv_fatbin``
     - 可执行文件
     - 说明
   * - ptx_only
     - 520 B
     - 1,023 KB
     - 独立 ``vector_add.fatbin`` 仅 504 B
   * - single
     - 5,592 B
     - 1,052 KB
     - PTX + cubin + dlink
   * - dual_no_ptx
     - 9,952 B
     - 1,057 KB
     - sm_75 + sm_89 双 cubin
   * - dual_ptx
     - 10,440 B
     - 1,057 KB
     - 相对 dual 仅 +488 B PTX

PTX-only 可执行文件比 single **小约 29 KB**——主要少在 cubin 机器码与
相关元数据。代价是 **首次运行** 必须支付 JIT 成本。

--------------

运行时：驱动如何选择 image？
--------------------------------

.. mermaid:: ../_static/ptx_jit_flow.mmd

简化决策（在 ``__cudaRegisterFatBinary`` 内，见 :doc:`../chapter_02_toolchain/11_register_chain`）：

1. 读取当前 GPU 计算能力，本机为 **sm_89**。
2. 在 fatbin 中查找 ``kind=elf`` 且 ``sm`` **精确匹配** 的 cubin。
3. 找到 → **直接映射 SASS**，不 JIT。
4. 未找到 → 查找 ``kind=ptx``；有则 **JIT** 为当前硬件 SASS。
5. 均无 → 模块注册失败。

本机验证：

.. list-table::
   :header-rows: 1
   :widths: 24 36

   * - 配置
     - sm_89 本机行为
   * - single / dual / dual_ptx
     - 选中 sm_89 cubin，``Max error: 0.000000``，**不走 JIT**
   * - ptx_only
     - 无 cubin，**必须 JIT**；结果仍正确
   * - dual_no_ptx
     - sm_89 仍用 cubin；若在 **无 cubin 的新架构** 上运行则失败

``dual_no_ptx`` 在 sm_90 等新一代 GPU 上 **无法** 靠 JIT 启动——这是
:doc:`04_multiarch_fatbinary` 建议 dual_ptx 的原因。

--------------

JIT 发生在哪里？strace 证据
--------------------------------

**关键对比**：离线 nvcc 与运行时 JIT 的进程模型完全不同。

.. list-table::
   :header-rows: 1
   :widths: 22 38 38

   * - 路径
     - execve
     - ComputeCache
   * - nvcc 离线 ``-arch=sm_89``
     - ``cicc``、``ptxas``、``fatbinary``
     - 不涉及
   * - fatbin 含 cubin，sm 精确匹配
     - 无
     - 通常不访问；实测 ``va2_single`` 零次打开
   * - fatbin 仅 PTX，首次运行
     - 无
     - cache miss：``O_WRONLY|O_CREAT`` 写入新条目
   * - 同一 PTX 第二次运行
     - 无
     - cache hit：``O_RDONLY`` 读已有缓存

PTX-only 首次 JIT 的 ``strace`` 片段（新 kernel ``vector_add2``，缓存未命中）：

.. code:: text

   openat(..., "/home/gzz/.nv/ComputeCache/index", O_RDWR) = 40
   openat(..., ".../e0943aa8a566c2", O_RDONLY) = -1 ENOENT
   openat(..., ".../e0943aa8a566c2", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 41

第二次运行同二进制：

.. code:: text

   openat(..., ".../e0943aa8a566c2", O_RDONLY) = 41    ← 直接命中

缓存目录结构为 ``ComputeCache/<hex>/<hex>/<hash>``，由 PTX 内容与目标 SM
等参数哈希决定。**驱动升级** 或 **PTX 变更** 会导致 hash 变化，旧缓存
自然失效。

相关环境变量：

.. list-table::
   :header-rows: 1
   :widths: 28 42

   * - 变量
     - 作用
   * - ``CUDA_CACHE_DISABLE=1``
     - 禁用磁盘缓存，每次强制 JIT
   * - ``CUDA_CACHE_PATH``
     - 自定义缓存根目录，默认 ``~/.nv/ComputeCache``
   * - ``CUDA_CACHE_MAXSIZE``
     - 缓存容量上限，字节

JIT 与 cubin 直载的 ioctl 模式在 **kernel launch** 阶段趋于一致；差异主要在
**模块注册** 阶段——PTX 路径需额外 GPU 交互分配代码段，见
:doc:`../chapter_04_runtime_advanced/03_module_loading` 对 ``ioctl(0x2a/0x2b)`` 的描述。

--------------

三种 JIT 入口对照
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - 入口
     - 输入
     - 触发时机
     - 典型场景
   * - 静态 fatbin
     - ``.nv_fatbin`` 段
     - ``__cudaRegisterFatBinary``
     - 仅 PTX 或 sm 不匹配
   * - ``cuModuleLoadData``
     - 外部 ``.ptx`` 文件
     - 显式 Driver API 加载
     - 插件、热更新 kernel
   * - nvcc 离线
     - ``.cu`` 源码
     - 编译期 ``execve ptxas``
     - 发布默认路径，零运行时 JIT

三条路径的 **JIT 引擎** 同属驱动内置 NVVM/ptxas 兼容层，但 **只有 nvcc 离线**
会 fork 子进程。静态 fatbin 与 ``cuModuleLoadData`` 均在 ``libcuda`` 内完成。

--------------

工程建议
--------------

1. **发布二进制**：目标架构用 ``code=sm_XX`` 预编译 cubin；对 **尚未发布的
   最高架构** 追加 ``code=compute_XX`` 作 PTX 回退，参考 dual_ptx。
2. **本机开发**：``-arch=native`` 或 CMake ``CUDA_ARCHITECTURES`` 与部署 GPU
   一致时，运行时几乎总走 cubin 直载。
3. **排查首次启动慢**：对可疑程序 ``strace -e openat`` 看 ComputeCache 是否
   ``O_CREAT``；或临时 ``CUDA_CACHE_DISABLE=1`` 对比。
4. **勿混淆**：fatbin 里 **带有 PTX** 不等于 **每次运行都 JIT**——sm 精确匹配
   cubin 时 PTX 仅作备用，不会被编译。

--------------

*分析基于 CUDA 13.1 (build 37061995)，实验目录 /tmp/ptx_jit_demo 与
``examples/vector_add.cu`` 互证。*

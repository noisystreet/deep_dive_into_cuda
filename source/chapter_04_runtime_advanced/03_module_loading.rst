cuModule 动态加载：Driver API 的运行时编译路径
====================================================

   第 1 章和第 2 章分析了 nvcc 的**离线编译**。``cuModuleLoadData``
   提供了完全不同的路径：在运行时直接从内存加载 cubin 或 PTX，无需
   调用 nvcc 子进程。本节对比三种 kernel 加载方式的底层行为差异。

   静态 fatbin 的 PTX JIT 回退见 :doc:`../chapter_01_compilation/06_ptx_jit`；
   本节聚焦 **Driver API 显式加载** 的另一条入口。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) / Linux x86-64

   测试程序: ``examples/module_demo.cu`` (三种加载路径)

--------------

三种加载路径概述
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - 路径
     - 入口
     - cuModuleLoad 输入
     - 编译时机
   * - **静态链接**
       (nvcc)
     - ``<<<>>>`` 语法
     - 第 1 章 / fat binary 嵌入
     - **编译期** nvcc execve
       子进程
   * - **动态 cubin**
       (Driver API)
     - ``cuModuleLoadData``
     - 独立编译的 ELF cubin
       (3.6 KB)
     - **运行时** 直接加载
       预编译 SASS
   * - **PTX JIT**
       (Driver API)
     - ``cuModuleLoadData``
     - 独立编译的 PTX 文本
       (1.3 KB)
     - **运行时** 进程内
       NVVM JIT 编译

离线编译的子进程问题
-----------------------

nvcc 的 ``-cubin`` 编译和普通 ``-o`` 编译一样，通过 ``execve`` 启动
子工具链：

::

   $ strace -f -e execve ./module_demo 2>&1 | grep -E 'nvcc|ptxas|cicc|fatbinary'

   execve("/usr/local/cuda/bin/nvcc", ...)    ← 主 nvcc 进程
   execve("gcc", ...)                         ← 主机端预处理
   execve("cudafe++", ...)                    ← CUDA 语法解析
   execve("cicc", ...)                        ← device 端编译 PTX
   execve("ptxas", ...)                       ← PTX → SASS
   execve("fatbinary", ...)                   ← 打包 fat binary
   execve("gcc", ...)                         ← 编译 host stub
   execve("g++", ...)                         ← 主机链接

但 ``cuModuleLoadData`` **不会产生任何 execve**。一旦 cubin 或 PTX
已被 nvcc 提前编译好，运行时加载完全是进程内操作。

cuModuleLoadData cubin：纯用户态加载
--------------------------------------

加载预编译的 ELF cubin 时，strace 显示只涉及文件操作和少量 ioctl：

.. code:: text

   openat(AT_FDCWD, "/tmp/dyn_kernel.sm_89.cubin", O_RDONLY) = 18
   read(18, image, 3624)                              ← 读取 cubin 到内存
   close(18)
   ; cuModuleLoadData — 在此处解析 ELF cubin
   ; cuModuleGetFunction — 按符号名查找 kernel 入口
   ; 以上两部均无 ioctl，无 execve

然后 kernel launch 的 ioctl 模式与普通 launch 无差异——加载完成后，
模块就是一个可调用的函数句柄：

.. code:: text

   ioctl(0x4e) × 1   ← cuLaunchKernel (与 <<<>>> 完全一致)

**核心发现：cuModuleLoadData 加载 cubin 不涉及内核交互**。cubin 一旦
加载，后续 launch 路径与静态编译的 kernel 完全一样。

PTX JIT：进程内编译的 ioctl 代价
------------------------------------

加载 PTX 时，``cuModuleLoadData`` 需要触发 PTX→SASS 的 JIT 编译。
这个编译在 **libcuda 进程内部** 完成（调用内置 NVVM 引擎），不产生
子进程。但 strace 能观察到 JIT 的间接成本。

与直接加载 cubin 相比，PTX JIT 路径的明显差异有：

.. code:: text

   ; 加载 PTX (cuModuleLoadData)
   ; 内部触发 JIT 编译，strace 表现：
   ioctl(0x4e)  × 2  ← JIT 过程中的 GPU 交互（内存分配等）
   ioctl(0x2a)  × 1  ← 额外的显存分配
   ioctl(0x2b)  × 1  ← fence

而在 cubin 加载场景下，这些 ioctl 都不存在——因为编译已在 nvcc 子
进程中完成，运行时只需解析 ELF 结构。

对比总结
----------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - 维度
     - 静态链接 (nvcc)
     - 动态 cubin
       (Driver API)
     - PTX JIT
       (Driver API)
   * - execve 子进程
     - **多次** (cicc/ptxas/...)
     - 无 (仅加载)
     - 无 (仅加载)
   * - ioctl 额外开销 (加载)
     - 无（编译时已支付）
     - 无
     - **少量** (JIT 内存分配)
   * - ioctl 额外开销 (launch)
     - 无（与标准相同）
     - 无
     - 无
   * - 部署方式
     - 单一可执行文件
     - 主程序 + cubin 文件
     - 主程序 + PTX 文件
   * - 跨架构兼容
     - fat binary 多架构
     - 不兼容（固定 sm_XX）
     - 兼容（JIT 编译）
   * - 调试 / 热更新
     - 需重新编译整个程序
     - 可独立替换 cubin
     - 可替换 PTX 文件

关键发现
-----------

1. **cuModuleLoadData 不启动 nvcc** — 与 Graph 实例化一样，cubin/PTX
   的运行时加载完全在 libcuda 进程内完成。这与 nvcc 的 execve 子进程
   模型形成鲜明对比。

2. **cubin 加载是纯用户态操作** — 加载预编译的 ELF cubin 不涉及
   ioctl，成本主要是文件读取 + ELF 解析。cuModuleGetFunction 按符号
   名查找 kernel 入口也是纯用户态。

3. **PTX JIT 有隐性 ioctl 成本** — 虽然 JIT 编译在进程内完成（无
   execve），但编译本身需要 GPU 交互（临时显存分配、代码段上传），
   这些会产生额外的 ioctl。这是 PTX 可移植性的代价。

4. **三条编译路径的进程模型** — CUDA 有三套完全不同的编译模型：

   - **nvcc 子进程**：离线编译，多次 execve
   - **PTX JIT**：进程内 NVVM，无 execve，少量 ioctl
   - **Graph JIT**：进程内 NVVM（与 PTX JIT 同一引擎），无 execve

5. **动态加载的部署灵活性** — cubin 动态加载允许在 **不修改主程序的
   情况下替换 GPU kernel**，这对算法迭代和热更新非常有用。代价是需要
   管理多个文件。

*分析基于 CUDA 13.1 / Driver 595.58.03。动态加载的 cubin 由 nvcc -cubin 独立编译。*

g++ 链接阶段：Fat Binary 如何进入可执行文件
=================================================

   nvcc 完成 device 编译后，产物是一个带有特殊段的 host 目标文件
   （``vector_add.cu.o``）。CMake / Makefile 的最后一步交给系统链接器
   ``g++``，将 fat binary 数据与 ``libcudart_static`` 等合并为 ELF 可执行文件。

   本节基于 ``examples/vector_add`` 的 CMake 构建日志与 ``readelf`` 实测。

   环境: CUDA 13.1 / sm_89 / g++ 14 (Debian)

--------------

两阶段链接概览
----------------

CMake 对 ``add_executable(vector_add vector_add.cu)`` 的处理分为两步：

.. mermaid:: ../_static/host_link.mmd

阶段 A：nvcc -c（device + host 合并在同一目标文件）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   /usr/local/cuda/bin/nvcc ... -x cu -c vector_add.cu \
       -o CMakeFiles/vector_add.dir/vector_add.cu.o

内部仍执行 cicc → ptxas → fatbinary → gcc 编译 ``vector_add.fatbin.c`` 等子步骤
（见第 1 章编译流水线），最终输出单个 ``vector_add.cu.o``。

阶段 B：g++ 链接（仅 host 链接器）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   /usr/bin/g++ -Wl,--dependency-file=.../link.d \
       @objects1.rsp -o vector_add @linkLibs.rsp \
       -L/usr/local/cuda/targets/x86_64-linux/lib ...

``objects1.rsp`` 内容（单文件工程）：

::

   CMakeFiles/vector_add.dir/vector_add.cu.o

``linkLibs.rsp`` 内容：

::

   -lcudadevrt -lcudart_static -lrt -lpthread -ldl

与第 1 章手工 nvcc 链路相比，CMake 3.18+ 的 CUDA 语言模式
把 device link（nvlink）藏进了 nvcc 的 -c 阶段 / 内部规则，最终由 g++
只看到一个 cu.o 目标文件；多 cu 源文件时 objects1.rsp 会列出多个
目标文件，并可能额外包含 dlink.o（见 :doc:`05_nvlink`）。

--------------

vector_add.cu.o 中的 CUDA 专用段
------------------------------------

``readelf -SW vector_add.cu.o`` 提取的关键段：

.. list-table::
   :header-rows: 1
   :widths: 22 10 28

   * - 段名
     - 大小
     - 说明
   * - ``.text``
     - 2015 B
     - host 代码（main、API 调用）
   * - ``__nv_module_id``
     - 15 B
     - 模块 ID 字符串
   * - ``.nv_fatbin``
     - 4064 B
     - fat binary 原始容器
   * - ``.nvFatBinSegment``
     - 24 B
     - ``__fatBinC_Wrapper_t``
   * - ``.rela.nvFatBinSegment``
     - 24 B
     - 指向 ``fatbinData`` 的重定位

符号表（``nm vector_add.cu.o``）：

::

   r fatbinData
   r _ZL15__module_id_str
   U __cudaRegisterFatBinary
   U __cudaRegisterFatBinaryEnd
   U __cudaUnregisterFatBinary

``U`` 表示未定义符号——注册 fat binary 的实现来自
``libcudart_static.a``，在 g++ 链接阶段解析。

--------------

fatbinary → C 数组 → 汇编段
-------------------------------

``vector_add.fatbin.c`` 在 ``--keep`` 模式下保留，其结构为：

1. 包含 ``fatbinary_section.h`` — 定义 FATBINC_MAGIC 与段名
2. 内联汇编写入 ``.nv_fatbin`` 段 — 508 个 ``.quad`` 指令填充 ``fatbinData``
3. ``__fatBinC_Wrapper_t`` 放入 ``.nvFatBinSegment``：

.. code:: c

   static const __fatBinC_Wrapper_t __fatDeviceText
     __attribute__((aligned(8)))
     __attribute__((section(".nvFatBinSegment")))
     = { 0x466243b1, 1, fatbinData, 0 };

``fatbinary_section.h`` 中的控制结构：

::

   #define FATBINC_MAGIC   0x466243B1    /* "FatBC" 小端 */
   #define FATBIN_CONTROL_SECTION_NAME  ".nvFatBinSegment"
   #define FATBIN_DATA_SECTION_NAME     ".nv_fatbin"

设计意图：数据段 nv_fatbin 与控制段 nvFatBinSegment 分离，
运行时可通过扫描 wrapper 快速定位 fat binary 指针，无需在 text 段中
硬编码偏移。

--------------

g++ 链接后 ELF 布局
---------------------

可执行文件 examples/build/vector_add（约 1.0 MB）段布局变化：

.. list-table::
   :header-rows: 1
   :widths: 20 12 30

   * - 段
     - 大小
     - 相对 cu.o 的变化
   * - ``.text``
     - 535,998 B
     - 合并 ``libcudart_static`` 等
   * - ``.rodata``
     - 57,888 B
     - 含 runtime 字符串表
   * - ``__nv_module_id``
     - 15 B
     - 不变
   * - ``.nv_fatbin``
     - 4064 B
     - 内容与 cu.o 中相同（直接合并）
   * - ``.nvFatBinSegment``
     - 24 B
     - 重定位后 data 指针指向 VA 0x9a230

Program Header 显示 nv_fatbin 段位于只读 LOAD 段（与 rodata 同段），
进程以 mmap 只读方式映射，符合「device 代码不可被 host 修改」的模型。

段 nv_fatbin 的头部与独立 ``vector_add.fatbin`` 文件一致（魔数 ba55ed50）：

::

   readelf -x .nv_fatbin vector_add | head
   9a230 50ed55ba 01001000 d00f0000 00000000  P.U.............

--------------

静态注册：从 .init_array 到 Driver
------------------------------------

链接 ``libcudart_static.a`` 后，可执行文件包含 CUDA 注册逻辑：

1. 编译器生成 cuda 注册构造函数（符号名含 sti 与 cudaRegisterAll，位于 ``vector_add.cu.o`` 的 text 段）
2. 指针写入 init_array 段（readelf 可见条目指向 VA 0x9329）
3. 进程启动时 glibc 调用该 constructor，链式执行：

.. code:: text

   __cudaRegisterFatBinary(&__fatDeviceText)
     → __cudaRegisterFatBinaryEnd(...)
     → __cudaRegisterFunction(...)   // 注册 kernel 入口

4. __cudaRegisterFatBinary 定义在 libcudart_static.a（T 符号，约 0x1da30）

反汇编上述注册函数可见对 __cudaRegisterFatBinary 与
__cudaRegisterFatBinaryEnd 的 call 指令，并将 handle 存入
__cudaFatCubinHandle（BSS）。

这与第 3 章 Kernel Launch 分析衔接：constructor 在 ``main`` 之前
完成 module 注册，``cudaLaunchKernel`` 才能通过 handle 找到 nv_fatbin 段内的 cubin。

--------------

链接库分工
------------

.. list-table::
   :header-rows: 1
   :widths: 22 40

   * - 库
     - 在链接阶段的作用
   * - ``libcudart_static``
     - Runtime API、cudaRegister 系列、cudaLaunchKernel stub
   * - ``libcudadevrt``
     - Device-side runtime（device 侧辅助符号；单 TU 仍链接）
   * - ``librt`` / ``libpthread``
     - 静态 cudart 依赖的 POSIX 接口
   * - ``libdl``
     - 动态加载 Driver（``libcuda.so``）

``-Wl,--start-group`` 与 ``--end-group`` 在手工 nvcc 链接时出现，
用于解决 cudart / cudadevrt 之间的循环引用；CMake 生成的 ``link.txt``
未使用该选项，但单文件 ``vector_add`` 链接仍成功——依赖顺序已由
``linkLibs.rsp`` 固定。

--------------

与 Fat Binary 文档的交叉引用
--------------------------------

Fat binary 容器格式与 image 目录结构见
:doc:`../chapter_01_compilation/03_fatbinary`。
本节回答的是：容器如何作为 ELF 段落地盘，以及 g++ 如何把它与 cudart 粘合成可执行文件。

--------------

*分析基于 CUDA 13.1 (build 37061995)，链接命令来自 examples/build/CMakeFiles/vector_add.dir/link.txt。*

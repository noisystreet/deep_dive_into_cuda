RDC 与可分离编译
=====================

   **Relocatable Device Code (RDC)** 允许多个 ``.cu`` 编译单元各自生成
   **可重定位的 device 代码**，再由 **nvlink** 在链接期解析跨 TU 的
   ``__device__`` / ``__global__`` 符号引用。

   单文件 ``examples/vector_add.cu`` 走 **Whole Program Compilation (WPC)**：
   device 符号在 ptxas 阶段即已解析完毕。本节用 ``examples/rdc/`` 两文件样例
   对比两种模式在段布局、fatbinary 调用与 g++ 链接对象数量上的差异。

   环境: CUDA 13.1 / sm_89 / Linux x86-64

--------------

为何需要 RDC
--------------

典型场景：库与主程序分文件

.. code:: cuda

   // device_utils.cu
   __device__ float rdc_add(float a, float b) { return a + b; }

   // vector_add_rdc.cu
   __device__ float rdc_add(float a, float b);
   __global__ void vector_add(...) { c[idx] = rdc_add(a[idx], b[idx]); }

若 **不** 开启 RDC，第二个 TU 在 ptxas 阶段无法解析 ``rdc_add``：

::

   ptxas fatal   : Unresolved extern function '_Z7rdc_addff'

开启 ``-rdc=true`` 后，每个 TU 输出 **带重定位信息的 device ELF**，
nvlink 将 ``device_utils.o`` 与 ``vector_add_rdc.o`` 链接为单一
``*_dlink.sm_89.cubin``，再经 fatbinary + link.stub 注册到运行时。

--------------

关键 nvcc 选项
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 38

   * - 选项
     - 含义
   * - ``-rdc=true`` / ``--relocatable-device-code=true``
     - 生成可重定位 device 代码（默认 false = WPC）
   * - ``-rdc=false``
     - Whole Program Compilation；单 TU 内符号必须在本 TU 解析
   * - ``-dc`` / ``--device-c``
     - 仅编译 device 对象（不完成 host 链接），输出仍含 ``__nv_relfatbin``
   * - ``-dw`` / ``--device-w``
     - 编译 device 对象并生成 host 包装（少见）
   * - ``-dlink`` / ``--device-link``
     - 单独执行 device link 步骤
   * - ``--no-device-link`` / ``-nodlink``
     - 跳过 device link（需后续手动 nvlink）

CMake 等价写法：

.. code:: cmake

   add_executable(rdc_vector_add rdc/device_utils.cu rdc/vector_add_rdc.cu)
   set_target_properties(rdc_vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

``CUDA_SEPARABLE_COMPILATION ON`` 会在链接阶段自动插入 nvlink + dlink.o。

--------------

编译流水线对比
----------------

.. mermaid:: ../_static/rdc_flow.mmd

WPC：单 TU vector_add.cu（回顾）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   g++ ... vector_add_dlink.o vector_add.o -lcudadevrt ...

-  仅 1 个用户 ``.cu.o`` 参与 host 链接，另加 dlink.o
-  每个 TU 的 fatbinary 不带 ``--device-c``
-  目标文件内段名为 ``.nv_fatbin``，最终可执行文件可能合并 module 与 dlink 两段

RDC：两 TU examples/rdc/（本节实测）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nvcc --verbose -rdc=true device_utils.cu vector_add_rdc.cu -o rdc_vector_add``
完整命令树要点：

1. 每个 TU 各一次 fatbinary，带 ``--device-c``：

   ::

      fatbinary ... --embedded-fatbin="device_utils.fatbin.c" --device-c
      fatbinary ... --embedded-fatbin="vector_add_rdc.fatbin.c" --device-c

2. **nvlink 合并两个 relocatable object**：

   ::

      nvlink ... "device_utils.o" "vector_add_rdc.o" -lcudadevrt \
          -o "rdc_vector_add_dlink.sm_89.cubin" \
          --register-link-binaries="rdc_vector_add_dlink.reg.c"

3. 第二次 fatbinary（``-link``）+ link.stub → ``rdc_vector_add_dlink.o``

4. **g++ 链接三个用户侧对象**：

   ::

      g++ ... rdc_vector_add_dlink.o device_utils.o vector_add_rdc.o \
          -lcudadevrt -lcudart_static ... -o rdc_vector_add

与 WPC 的关键差异：**每个 ``.cu.o`` 都进入最终链接**，且 TU 级 fatbin 使用
``__nv_relfatbin`` 段而非直接 ``.nv_fatbin``。

--------------

段布局：__nv_relfatbin vs .nv_fatbin
---------------------------------------

``readelf -SW`` 实测（sm_89）：

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 28

   * - 文件
     - 段名
     - 大小
     - 说明
   * - device_utils.o (RDC)
     - ``__nv_relfatbin``
     - 1072 B
     - 可重定位 fatbin 数据
   * - vector_add_rdc.o (RDC)
     - ``__nv_relfatbin``
     - 1760 B
     - 含未解析 ``rdc_add`` 的 cubin
   * - rdc_vector_add_dlink.o
     - ``.nv_fatbin``
     - 5304 B
     - 链接后最终 device 镜像
   * - vector_add.o (WPC)
     - ``.nv_fatbin``
     - 4064 B
     - 模块级 PTX+cubin（非 RDC）

最终可执行文件 ``rdc_vector_add``：

::

   [18] .nv_fatbin        5304 B   ← dlink 产物（运行时注册主 fatbin）
   [19] __nv_relfatbin    2832 B   ← 1072 + 1760，两 TU 的 relocatable 数据合并
   [38] .nvFatBinSegment   72 B   ← 多个 wrapper（含 dlink + 各 TU）

WPC 的 ``vector_add`` 则通常将 module 与 dlink 的 ``.nv_fatbin`` **合并为一段**
（见 :doc:`07_host_link`、:doc:`08_fatbinary`），且 **无** ``__nv_relfatbin``。

--------------

nvlink 与 reg.c：跨 TU 符号解析
----------------------------------

链接后 cubin（``cuobjdump -symbols``）可见两个全局 device 符号均已解析：

::

   _Z7rdc_addff                              ← device_utils.cu
   _Z10vector_addPKfS0_Pfi                   ← vector_add_rdc.cu

``rdc_vector_add_dlink.reg.c`` 内容：

.. code:: c

   #define NUM_PRELINKED_OBJECTS 2
   DEFINE_REGISTER_FUNC(_cc4b6d89_15_device_utils_cu_20c8e3a2)
   DEFINE_REGISTER_FUNC(_48231fa4_17_vector_add_rdc_cu_052d2be5)

``NUM_PRELINKED_OBJECTS`` 等于 **参与 device link 的 TU 数量**。
``link.stub`` 编译为 ``dlink.o`` 后，在 ``__cudaRegisterLinkedBinary*`` 中
注册 **链接后的** fatbin，并关联各 TU 的预链接元数据。

``vector_add_rdc.o`` 的未定义符号在 device link 前包括：

::

   U __cudaRegisterLinkedBinary_48231fa4_17_vector_add_rdc_cu_052d2be5

--------------

fatbinary 调用次数
--------------------

对两 TU RDC 工程，``nvcc --verbose`` 中共 **3 次** fatbinary：

.. list-table::
   :header-rows: 1
   :widths: 8 20 32

   * - 次序
     - 输出
     - 标志
   * - 1
     - device_utils.fatbin.c
     - ``--device-c``，relocatable
   * - 2
     - vector_add_rdc.fatbin.c
     - ``--device-c``
   * - 3
     - rdc_vector_add_dlink.fatbin.c
     - ``-link``，链接 cubin

第 1、2 次对应 :doc:`08_fatbinary` 中「模块级」打包，但 ``--device-c`` 使
输出进入 ``__nv_relfatbin`` 段；第 3 次与 WPC 单 TU 的 dlink fatbinary 相同。

--------------

体积与性能侧记
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 14

   * - 产物
     - 大小
   * - device_utils.o
     - 5672 B
   * - vector_add_rdc.o
     - 11640 B
   * - rdc_vector_add_dlink.o
     - 9848 B
   * - rdc_vector_add_dlink.sm_89.cubin
     - 5224 B
   * - rdc_vector_add（可执行文件）
     - 1,058,744 B

RDC 增加 nvlink 与额外 fatbinary/stub 步骤，可执行文件略大于 WPC 单 TU 工程；
收益在于 **多文件工程的可维护性** 与 **device 库复用**，而非体积。

运行验证：

::

   $ ./rdc_vector_add
   RDC vector_add max error: 0.000000

--------------

何时启用 RDC
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 38

   * - 场景
     - 建议
   * - 单 ``.cu`` 小程序
     - ``-rdc=false``，默认 WPC
   * - 多 ``.cu`` + 跨 TU ``__device__`` 函数
     - ``-rdc=true`` + nvlink
   * - 静态 device 库（``libcudadevrt`` 以外自定义）
     - 通常 ``-rdc=true`` 编译库，最终 app 链接时再 device link
   * - 仅 host 调用、device 全在一个 TU
     - 不必 RDC

--------------

与其他章节的边界
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 32

   * - 主题
     - 章节
   * - nvlink 命令行与 LTO
     - :doc:`05_nvlink`
   * - fatbinary ``--device-c`` / ``-link``
     - :doc:`08_fatbinary`
   * - g++ 链接对象列表
     - :doc:`07_host_link`
   * - RDC 段布局与 reg.c
     - **本节**

--------------

*分析基于 CUDA 13.1 (build 37061995)，样例 ``examples/rdc/`` 与 /tmp/rdc_demo 构建日志互证。*

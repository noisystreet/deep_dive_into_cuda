NVCC 编译过程分析
=================

   基于 ``nvcc --verbose -o vector_add vector_add.cu`` 的实际编译日志
   (CMake 驱动的 NVCC verbose 输出)

   环境: CUDA 13.1, GPU Architecture: sm_89 (Ada Lovelace), Host: x86_64
   Linux

概述
----

NVCC (NVIDIA CUDA Compiler) 将 ``.cu``
文件编译为可执行文件的过程远比普通 C++
编译器复杂。它需要处理两类代码——**Host 代码**\ （运行在 CPU 上）和
**Device 代码**\ （运行在 GPU
上），并最终将它们链接到同一个可执行文件中。

整个过程分为 **11 个步骤**\ ，可归纳为 **5 个阶段**\ ：

1. **环境准备** — 设置编译环境变量
2. **Host 前端编译** — 预处理 + CUDA 语法解析
3. **Device 编译** — cicc(PTX) → ptxas(cubin) → fatbinary(嵌入)
4. **Host 后端编译** — 将翻译后的 host 代码编译为 ``.o``
5. **链接** — Device 链接(nvlink) + Host 链接(g++)

下面逐阶段详细分析。

--------------

阶段 1: 环境准备
----------------

编译开始前，NVCC 打印出所有环境配置：

.. code:: bash

   #$ _NVVM_BRANCH_=nvvm
   #$ _CUDART_=cudart
   #$ _HERE_=/usr/local/cuda/bin
   #$ _THERE_=/usr/local/cuda/bin
   #$ TOP=/usr/local/cuda/bin/..
   #$ CICC_PATH=/usr/local/cuda/bin/../nvvm/bin
   #$ NVVMIR_LIBRARY_DIR=/usr/local/cuda/bin/../nvvm/libdevice
   #$ INCLUDES="-I/usr/local/cuda/bin/../targets/x86_64-linux/include"
   #$ SYSTEM_INCLUDES="-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"
   #$ LIBRARIES="-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" \
   #            "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"

这些变量定义了编译工具链的路径：

+-----------------------+-----------------------+-----------------------+
| 变量                  | 路径                  | 说明                  |
+=======================+=======================+=======================+
| ``TOP``               | ``/us                 | CUDA 安装根目录       |
|                       | r/local/cuda/bin/..`` |                       |
+-----------------------+-----------------------+-----------------------+
| ``CICC_PATH``         | ``$TOP/nvvm/bin``     | CUDA device 编译器    |
|                       |                       | (cicc) 位置           |
+-----------------------+-----------------------+-----------------------+
| `                     | ``                    | NVVM IR               |
| `NVVMIR_LIBRARY_DIR`` | $TOP/nvvm/libdevice`` | 库（内置数学函数等）  |
+-----------------------+-----------------------+-----------------------+
| ``INCLUDES``          | ``$TOP/targets/x      | CUDA 头文件路径       |
|                       | 86_64-linux/include`` |                       |
+-----------------------+-----------------------+-----------------------+
| ``LIBRARIES``         | ``$TOP/targe          | CUDA 库文件路径       |
|                       | ts/x86_64-linux/lib`` |                       |
+-----------------------+-----------------------+-----------------------+

关键预定义宏： - ``__CUDA_ARCH_LIST__=890`` — 目标 GPU 架构为 sm_89 -
``__CUDACC_VER_MAJOR__=13``, ``__CUDACC_VER_MINOR__=1``,
``__CUDACC_VER_BUILD__=115`` — CUDA 版本 13.1 - ``__CUDACC__``,
``__NVCC__`` — 标识当前为 NVCC 编译环境

--------------

阶段 2: Host 前端编译（预处理 + CUDA 语法解析）
-----------------------------------------------

步骤 1: Host 预处理
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   gcc -D__CUDA_ARCH_LIST__=890 -E -x c++ -D__CUDACC__ -D__NVCC__ \
       "-I/usr/local/cuda/bin/../targets/x86_64-linux/include" \
       "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl" \
       -D__CUDACC_VER_MAJOR__=13 ... -include "cuda_runtime.h" \
       -m64 "vector_add.cu" \
       -o "/tmp/tmpxft_...-5_vector_add.cpp4.ii"

-  **输入**: ``vector_add.cu``
-  **工具**: ``gcc`` (作为预处理器)
-  **输出**: ``.cpp4.ii`` (预处理后的 C++ 源文件)
-  **作用**: 展开所有 ``#include``\ 、宏定义、条件编译。注意这里使用了
   ``-include "cuda_runtime.h"`` 强制包含 CUDA 运行时头文件。

步骤 2: CUDA 前端 (cudafe++)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cudafe++ --c++17 --static-host-stub --device-hidden-visibility \
       --gnu_version=140200 --display_error_number \
       --orig_src_file_name "vector_add.cu" \
       --orig_src_path_name "/home/user/creativity/deep_dive_into_cuda/src/vector_add.cu" \
       --allow_managed --m64 --parse_templates \
       --gen_c_file_name "/tmp/tmpxft_...-6_vector_add.cudafe1.cpp" \
       --stub_file_name "tmpxft_...-6_vector_add.cudafe1.stub.c" \
       --gen_module_id_file \
       --module_id_file_name "/tmp/tmpxft_...-4_vector_add.module_id" \
       "/tmp/tmpxft_...-5_vector_add.cpp4.ii"

-  **输入**: ``.cpp4.ii`` (预处理后的文件)
-  **工具**: ``cudafe++``
-  **输出**:

   -  ``.cudafe1.cpp`` — 翻译后的 host C++ 代码（device 代码被替换为
      stub/调用）
   -  ``.stub.c`` — device 函数的 stub 声明
   -  ``.module_id`` — 模块 ID

-  **作用**: **这是 CUDA 编译最核心的步骤之一**\ 。\ ``cudafe++`` 解析
   CUDA 语法（\ ``__global__``\ 、\ ``<<<...>>>`` 等），将 device 代码与
   host 代码分离，并生成对应的 host stub。Host 代码中原本的 kernel 调用
   ``vector_add<<<blocks, threads>>>()`` 会被替换为
   ``__cudaLaunchKernel()`` 运行时调用，而 device 代码
   ``__global__ void vector_add(...)`` 则被提取出来交由后面的 device
   编译器处理。

--------------

阶段 3: Device 编译路径
-----------------------

步骤 3: Device 预处理
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   gcc -D__CUDA_ARCH__=890 -D__CUDA_ARCH_LIST__=890 -E -x c++ \
       -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__ \
       ... -include "cuda_runtime.h" -m64 "vector_add.cu" \
       -o "/tmp/tmpxft_...-9_vector_add.cpp1.ii"

-  **与步骤 1 的区别**: 增加了 ``-D__CUDA_ARCH__=890``\ ，标识 device
   代码目标架构为 sm_89，使 device 函数调用可以针对特定架构优化。

步骤 4: CUDA Device 编译器 (cicc) — 生成 PTX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   "$CICC_PATH/cicc" --c++17 --static-host-stub --device-hidden-visibility \
       --gnu_version=140200 --display_error_number \
       --orig_src_file_name "vector_add.cu" \
       --orig_src_path_name "/home/user/creativity/deep_dive_into_cuda/src/vector_add.cu" \
       --allow_managed -arch compute_89 -m64 --no-version-ident \
       -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 \
       --include_file_name "tmpxft_...-3_vector_add.fatbin.c" \
       -tused --module_id_file_name "...module_id" \
       --gen_c_file_name "...cudafe1.c" \
       --stub_file_name "...cudafe1.stub.c" \
       --gen_device_file_name "...cudafe1.gpu" \
       "/tmp/tmpxft_...-9_vector_add.cpp1.ii" \
       -o "/tmp/tmpxft_...-6_vector_add.ptx"

====================== ======================================
参数                   说明
====================== ======================================
``-arch compute_89``   目标虚拟架构为 compute_89（PTX 版本）
``-ftz=0``             不刷新非规格化数为 0
``-prec_div=1``        使用精确除法
``-prec_sqrt=1``       使用精确开方
``-fmad=1``            允许 FMA (fused multiply-add) 指令融合
``--no-version-ident`` 不在 PTX 中写入版本标识
====================== ======================================

-  **输入**: ``.cpp1.ii`` (device 预处理文件)
-  **工具**: ``cicc`` (CUDA Intermediate Code Compiler) — 基于 LLVM/NVVM
   的编译器
-  **输出**: ``.ptx`` (Parallel Thread Execution 汇编)
-  **作用**: 将 CUDA C++ device 代码编译为 **PTX 汇编**\ 。PTX
   是一种\ **虚拟指令集架构**\ （ISA），它独立于具体的 GPU
   硬件，提供了一层抽象。同一个 PTX 代码可以在不同代的 GPU
   上运行（只要驱动能将其翻译为目标硬件指令）。

浮点精度参数 ``-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1``
确保了数值计算的 IEEE 754 兼容性。

步骤 5: PTX 汇编器 (ptxas) — 生成 SASS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   ptxas -arch=sm_89 -m64 \
       "/tmp/tmpxft_...-6_vector_add.ptx" \
       -o "/tmp/tmpxft_...-10_vector_add.sm_89.cubin"

-  **输入**: ``.ptx`` (PTX 汇编)
-  **工具**: ``ptxas`` (PTX Assembler)
-  **输出**: ``.cubin`` (CUDA Binary) — 实际的 GPU 机器码 (SASS)
-  **作用**: 将虚拟指令 PTX **汇编为具体的 GPU 机器码 (SASS)**\ ，针对
   sm_89 架构（Ada
   Lovelace）进行指令调度和寄存器分配。这一步是性能关键——ptxas
   会做指令级优化。

步骤 6: fatbinary — 打包
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   fatbinary -64 \
       --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1" \
       "--image3=kind=elf,sm=89,file=...sm_89.cubin" \
       "--image3=kind=ptx,sm=89,file=...ptx" \
       --embedded-fatbin="/tmp/tmpxft_...-3_vector_add.fatbin.c"

-  **输入**: ``.cubin`` + ``.ptx``
-  **工具**: ``fatbinary``
-  **输出**: ``.fatbin.c`` (C 源文件，包含嵌入的二进制数据)
-  **作用**: 将 cubin(ELF) 和 PTX 打包成一个 “fat binary”。\ **之所以叫
   fat binary，是因为它可以包含针对多个 GPU 架构的代码**\ 。运行时 CUDA
   驱动会根据实际 GPU 选择最合适的二进制。这里同时包含了 PTX（JIT
   回退用）和 cubin（直接执行用）。
-  之后删除临时文件 ``rm ...fatbin``\ 。

--------------

阶段 4: Host 后端编译
---------------------

步骤 7: 编译 Host 代码
~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   gcc -D__CUDA_ARCH__=890 -D__CUDA_ARCH_LIST__=890 -c -x c++ \
       -DCUDA_DOUBLE_MATH_FUNCTIONS -Wno-psabi \
       "-I/usr/local/cuda/bin/../targets/x86_64-linux/include" \
       "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl" \
       -m64 "/tmp/tmpxft_...-6_vector_add.cudafe1.cpp" \
       -o "/tmp/tmpxft_...-11_vector_add.o"

-  **输入**: ``.cudafe1.cpp`` (cudafe++ 翻译后的 host 端 C++ 代码)
-  **工具**: ``gcc``
-  **输出**: ``vector_add.o`` (Host 目标文件)
-  **作用**: 标准的 C++ 编译。此时文件已经是纯 C++（CUDA
   扩展语法已被翻译为 CUDA 运行时 API 调用），所以可以直接用 gcc
   编译为目标文件。\ ``-Wno-psabi`` 用于抑制 ABI 警告。

--------------

阶段 5: 链接
------------

步骤 8: Device 链接 (nvlink)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   nvlink -m64 --arch=sm_89 \
       --register-link-binaries="..._dlink.reg.c" \
       "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" \
       "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" \
       -cpu-arch=X86_64 "...11_vector_add.o" \
       -lcudadevrt \
       -o "...12_vector_add_dlink.sm_89.cubin" \
       --host-ccbin "gcc"

-  **输入**: ``vector_add.o``\ （包含 device 代码的 cubin 段）
-  **工具**: ``nvlink`` (NVIDIA Linker)
-  **输出**: ``_dlink.sm_89.cubin`` (链接后的 device cubin) +
   ``_dlink.reg.c`` (device 注册代码)
-  **作用**: **Device 代码链接**\ 。如果程序包含多个 ``.cu``
   文件，各自的 device 代码需要被链接在一起，解析 device
   函数间的引用。即使这里只有一个 ``.cu`` 文件，\ ``nvlink`` 仍会处理
   ``-lcudadevrt`` (CUDA device runtime 库) 中 device 函数的链接。

步骤 9/10: 再次 fatbinary + 编译 link stub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 将链接后的 device cubin 再次打包为 fat binary
   fatbinary -64 --cicc-cmdline="..." -link \
       "--image3=kind=elf,sm=89,file=..._dlink.sm_89.cubin" \
       --embedded-fatbin="..._dlink.fatbin.c"

   # 编译 link stub
   gcc ... -DFATBINFILE="..._dlink.fatbin.c" \
       -DREGISTERLINKBINARYFILE="..._dlink.reg.c" \
       ... "/usr/local/cuda/bin/crt/link.stub" \
       -o "...13_vector_add_dlink.o"

-  **作用**: 将 nvlink 后的 device cubin 嵌入到 C 源文件中，然后编译为
   link stub 目标文件。这个 ``.o`` 文件包含了用于在运行时注册 device
   模块的代码。

步骤 11: 最终链接
~~~~~~~~~~~~~~~~~

.. code:: bash

   g++ -D__CUDA_ARCH_LIST__=890 -m64 \
       -Wl,--start-group \
       "...13_vector_add_dlink.o" \
       "...11_vector_add.o" \
       "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" \
       "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" \
       -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
       -Wl,--end-group \
       -o "vector_add"

-  **输入**: 所有 ``.o`` 文件 + CUDA 运行时库
-  **工具**: ``g++`` (系统 C++ 链接器)
-  **输出**: ``vector_add`` (最终可执行文件)
-  **链接的库**:

+-----------------------------------+-----------------------------------+
| 库                                | 说明                              |
+===================================+===================================+
| ``-lcudadevrt``                   | CUDA Device Runtime — 支持在      |
|                                   | device 代码中动态调用 kernel      |
+-----------------------------------+-----------------------------------+
| ``-lcudart_static``               | CUDA Runtime API (静态链接) —     |
|                                   | 提供                              |
|                                   | `                                 |
|                                   | `cudaMalloc``\ 、\ ``cudaMemcpy`` |
|                                   | 等 API                            |
+-----------------------------------+-----------------------------------+
| ``-lrt``                          | POSIX 实时扩展库 (clock_gettime   |
|                                   | 等)                               |
+-----------------------------------+-----------------------------------+
| ``-lpthread``                     | POSIX 线程库                      |
+-----------------------------------+-----------------------------------+
| ``-ldl``                          | 动态链接加载库 (dlopen 等，用于   |
|                                   | CUDA driver 加载)                 |
+-----------------------------------+-----------------------------------+

``-Wl,--start-group``/``--end-group``
确保这些静态库中的符号能够正确解析（处理循环依赖）。

--------------

完整流程总览
------------

.. mermaid:: ../_static/compilation_journey.mmd

两条编译路径
~~~~~~~~~~~~

编译过程中存在两条并行的路径：

+-----------------+-----------------+-----------------+-----------------+
| 路径            | 工具链          | 产物            | 目标            |
+=================+=================+=================+=================+
| **Host 路径**   | gcc → g++       | ``.o`` →        | CPU             |
|                 |                 | 可执行文件      | 上执行的代码    |
+-----------------+-----------------+-----------------+-----------------+
| **Device 路径** | cicc → ptxas →  | ``.ptx`` →      | GPU             |
|                 | fatbinary       | ``.cubin`` →    | 上执行的代码    |
|                 |                 | ``.fatbin.c``   |                 |
+-----------------+-----------------+-----------------+-----------------+

两条路径通过 fatbinary 和 nvlink 在链接阶段汇合——device 二进制数据以 C
数组的形式嵌入到 host 目标文件中，运行时由 CUDA driver 加载到 GPU 上。

关键设计思想
~~~~~~~~~~~~

1. **分离编译 (Split Compilation)**: CUDA 程序必须被拆分为 host 和
   device 两部分，分别用不同的编译器处理。
2. **PTX 虚拟 ISA**: 引入 PTX 作为中间层，使得同一份 CUDA
   代码可以兼容多代 GPU 硬件（编译为 PTX，运行时 JIT 编译为具体
   SASS，或直接使用预编译的 cubin）。
3. **Fat Binary**: 在一个可执行文件中打包多架构的 GPU
   代码，实现二进制兼容性。
4. **Two-stage linking**: device 代码需要独立的链接步骤 (nvlink)
   来处理跨编译单元的 device 函数调用。

--------------

源代码与编译
------------

.. mermaid:: ../_static/compilation_pipeline.mmd

*分析基于 CUDA 13.1 / sm_89 (Ada Lovelace)
架构，不同版本/架构的详细参数可能有所不同。*

术语表
=======

.. glossary::

   PTX
      Parallel Thread Execution，NVIDIA GPU 的**虚拟指令集架构（ISA）**。
      独立于具体 GPU 硬件，同一 PTX 可在不同代 GPU 上运行。
      PTX 提供无限虚拟寄存器，显式表达内存层次（global/shared/local），
      是理解 GPU 编程模型的最佳入口。

   SASS
      Streaming ASSembler，NVIDIA GPU 的**实际硬件机器码**。
      由 ``ptxas`` 将 PTX 汇编为针对特定 SM 架构（如 sm_89）的
      二进制指令，包含真实的寄存器分配与指令调度。

   cubin
      CUDA Binary，包含 SASS 机器码的 **ELF 格式可执行文件**。
      可直接由 CUDA 驱动加载到 GPU 执行。多个 cubin 可通过
      ``nvlink`` 链接为单一的 device 可执行镜像。

   fat binary
      包含多种 GPU 架构代码的 **容器格式**。一个 fat binary 可同时
      打包 PTX 文本和针对不同 SM 的 ELF cubin。运行时由驱动根据
      实际 GPU 型号选择最优版本。

   NVVM
      NVIDIA Virtual Machine，基于 LLVM 7.0.1 的编译框架。
      将 CUDA C++ device 代码降为 LLVM IR，运行定制 Pass 后，
      通过 NVPTX 后端输出 PTX。cicc 的上层包装调用 libnvvm.so
      中的 NVVM API。

   cicc
      CUDA Intermediate Code Compiler，将 CUDA C++ 编译为 PTX。
      基于 NVVM/LLVM 框架，内部通过 ``dlopen`` 加载 ``libnvvm.so``
      （61 MB）和 ``libdevice.10.bc`` （454 KB，内置数学函数）。

   ptxas
      PTX Assembler，将 PTX 编译为 SASS 机器码。**两阶段架构**：
      单线程 Backend/Optimizer（寄存器分配、指令调度）+ 多线程
      Finalizer（代码编码、二进制发射）。对于复杂 kernel，
      ``--split-compile`` 可并行加速。

   nvlink
      NVIDIA Linker，将多个 device 编译单元的 cubin 链接为单一
      ELF cubin。处理符号解析、重定位、段合并，并支持 LTO
      （链接时优化，调用 cicc + ptxas JIT 重新编译）。

   cudafe++
      CUDA Front End，基于 EDG C++ 6.7 的 CUDA 语法解析器。
      从预处理后的 ``.cpp4.ii`` 同时生成三个输出文件：
      host C++ 代码、device 代码、kernel host stub。

   SM
      Streaming Multiprocessor，GPU 中的**计算单元**。
      每个 SM 包含多个 CUDA Core、共享内存、寄存器文件和 warp
      scheduler。SM 数量决定了 GPU 的并行吞吐能力。

   BAR
      Base Address Register，PCIe 地址空间映射区域。
      GPU 显存通过 PCIe BAR 映射到 CPU 的物理地址空间，
      操作系统通过 ``mmap`` 将其暴露给用户态。

   ioctl
      设备驱动系统调用接口，用户态与内核态通信的通道。
      CUDA Driver API 最终通过 ``ioctl(NV_DEV_IOCTL(*))``
      与 nvidia.ko 内核模块交互，完成显存分配、kernel 提交、
      DMA 传输等硬件操作。

   WMMA
      Warp Matrix Multiply-Accumulate，NVIDIA 的 **Warp 级矩阵乘 API**。
      使用 ``nvcuda::wmma::mma_sync`` 在 PTX 层面映射为
      ``wmma.mma.sync`` 指令，ptxas 将其编译为 ``HMMA.16816.F32``
      Tensor Core 指令。一条 HMMA 覆盖 4096 个乘加操作。

   Tensor Core
      NVIDIA GPU 的**专用矩阵运算硬件单元**，自 Volta (sm_70) 起引入。
      原生支持混合精度矩阵乘法（如 half→float），吞吐远超 CUDA Core
      组合。通过 WMMA API 或 ``HMMA`` 指令直接调用。

   TileIR
      CUDA 13.1 引入的 **分块（Tile）级中间表示**，以 Tile IR 字节码
      描述「分块」级计算，由 ``tileiras`` 汇编为 SASS。与传统 SIMT
      路径（cicc → PTX → ptxas）并行存在，可嵌入 fat binary。

   UVM
      Unified Virtual Memory，CUDA 的统一虚拟内存模型。通过
      ``nvidia_uvm.ko`` 内核模块实现 GPU 和 CPU 共享统一地址空间，
      运行时透明处理 page fault 和数据迁移。

   eventfd
      Linux 内核的**事件通知机制**。CUDA 驱动使用 ``eventfd2`` 创建
      文件描述符，GPU 完成时通过写入 8 字节触发通知，
      用户态通过 ``futex`` 等待，实现从硬件中断到用户态同步的
      完整桥接。

   HMMA
      Half-precision Matrix Multiply-Accumulate，Tensor Core 的
      SASS 级原生指令。格式为 ``HMMA.16816.F32`` — 16×16×16 矩阵、
      half 输入 / float 累加。每条指令处理 4096 个乘加操作。

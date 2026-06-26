术语表
=======

.. glossary::

   PTX
      Parallel Thread Execution，NVIDIA GPU 的虚拟指令集架构（ISA）。

   SASS
      Streaming ASSembler，NVIDIA GPU 的实际硬件机器码。

   cubin
      CUDA binary，包含 SASS 机器码的 ELF 格式文件。

   fat binary
      包含多种 GPU 架构代码（PTX + SASS）的容器格式。

   NVVM
      NVIDIA Virtual Machine，基于 LLVM 的编译框架。

   cicc
      CUDA Intermediate Code Compiler，将 CUDA C++ 编译为 PTX。

   ptxas
      PTX Assembler，将 PTX 编译为 SASS 机器码。

   nvlink
      NVIDIA Linker，将多个 device 编译单元链接为单一 cubin。

   cudafe++
      CUDA Front End，基于 EDG C++ 的 CUDA 语法解析器。

   SM
      Streaming Multiprocessor，GPU 中的计算单元。

   BAR
      Base Address Register，PCIe 地址空间映射区域。

   ioctl
      设备驱动系统调用接口，用户态与内核态通信的通道。

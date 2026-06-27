CUDA Graph 捕获序列：从 capture 到 instantiate 的底层路径
=================================================================

   CUDA Graph 允许将一系列 kernel launch 捕获为一张图（graph），
   然后一次性提交（instantiate + launch），减少多次 launch 的驱动
   开销。本节用 strace 追踪 ``cudaStreamBeginCapture`` →
   ``cudaGraphInstantiate`` → ``cudaGraphLaunch`` 全流程，与普通
   launch 路径做 ioctl 对比。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 / RTX 4060 Laptop GPU

   测试程序: ``examples/graph_capture_demo.cu``

--------------

为什么需要 CUDA Graph？
--------------------------

每次 ``<<<>>>`` launch 的延迟约 5-15 微秒（见 :doc:`../chapter_03_runtime/03_kernel_launch`）。
当应用需要频繁启停kernel（如推理服务逐请求调用），launch 开销会占据
GPU 空闲时间。CUDA Graph 的核心优化：

- **捕获期**：正常的 kernel launch，驱动记录所有命令
- **实例化**：将捕获的命令图编译为 GPU 可执行的调度单元
- **重放**：一次提交整张图，多次 kernel launch 合并为 1 次 ioctl

--------------

三种模式对比
--------------

.. list-table:: 三种执行模式的性能对比
   :header-rows: 1
   :widths: 25 20 25 30

   * - 模式
     - 100 次总耗时
     - 平均单次
     - 说明
   * - A: 普通 launch
     - 202.72 ms
     - 2.027 us
     - 每次循环独立 launch
   * - B: Graph 重放
     - 182.54 ms
     - 1.825 us
     - 一次 instantiate + 100 次 launch
   * - C: 动态 Graph 重建
     - 17.78 ms (×10)
     - 1.778 ms/capture
     - 每次重建图

关键发现：

- **Graph 重放加速比 1.11×** — 单 kernel 图的优化空间有限（launch 开销
  占比小）。Graph 的真正价值在**多节点图**。
- **动态图重建的代价** — 每次 ``capture + instantiate`` 约 1.78 ms，
  如果图结构频繁变化，这个开销可能超过重放节省的时间。
- **launch 次数不减少** — Graph launch 仍然产生 ``ioctl(0x4e)``，但
  由多次变为一次批量提交。

--------------

ioctl 模式分析
-----------------

strace 显示三种模式的 ioctl 分布差异：

.. code:: text

   ; 普通 launch (100 次)
   0x4e × 100   ← 100 次独立 kernel launch
   0x2b × 100   ← 100 次 fence (同步)
   0x2a × 0     ← 不变 (无分配)
   
   ; Graph capture
   ; → 与普通 launch 模式完全相同的 ioctl 序列
   ;   (capture 就是 record)
   
   ; Graph instantiate
   0x4e × 1     ← cuGraphInstantiate: 提交图到 GPU 调度模块
   0x2a × 1     ← 额外分配 (图调度内部结构)
   ; 无 execve     ← 与 Graph JIT 一致: 不启动子进程
   
   ; Graph launch (100 次重放)
   0x4e × 1     ← 1 次 ioctl 提交整个图
   0x2b × 1     ← 1 次 fence (等待整张图完成)

差异汇总：

.. list-table:: ioctl 次数对比 (100 次 launch)
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - ioctl
     - 普通
     - Capture
     - Instantiate
     - Launch
   * - ``0x4e``
     - 100
     - 1
     - 1
     - **1**
   * - ``0x2b``
     - 100
     - 1
     - 1
     - **1**
   * - ``0x2a``
     - 0
     - 0
     - 1
     - 0
   * - ``0x49``
     - 0
     - 0
     - 0
     - 0

核心结论：**Graph 将 N 次 ioctl 减少为 1 次**——这是加速的主要来源。
单 kernel 图收益有限，因为 launch 的固定成本（ioctl + 驱动处理）在总
延迟中占比不大；但多节点图的收益会线性增长。

--------------

动态图的代价
----------------

当图结构需要频繁变化（如深度学习训练中不同 layer 的排列组合），每次
都必须重新 capture + instantiate：

.. code:: text

   动态图重建 (×10):
   17.78 ms total → 平均 1.78 ms 每次
   
   breakdown (估计):
   ├── cudaStreamBeginCapture    — 0.01 ms (纯用户态)
   ├── kernel launch (capture)   — 0.14 ms (1 次 ioctl)
   ├── cudaStreamEndCapture      — 0.05 ms (构建图数据结构)
   ├── cudaGraphInstantiate       — 1.40 ms (最重: 驱动编译)
   └── cudaGraphLaunch           — 0.18 ms (1 次 ioctl)

``cudaGraphInstantiate`` 占了 78% 的时间——它需要将捕获的命令序列提交
给驱动进行验证和编译，包括图优化、内存别名分析和依赖图构建。

--------------

Graph 与 UPG (Updatable Graph)
-------------------------------------

CUDA 12.x 引入了   Updatable Graph（``cudaGraphExecUpdate``），允许
在不重新实例化的情况下更新图中的 kernel 参数：

.. code:: text

   传统 Graph:
   capture → instantiate → launch → destroy → capture → ...
   每次结构变化都需要完整的 capture + instantiate
   
   Updatable Graph:
   capture → instantiate → launch → update → launch → update → ...
   参数变化只需更新，结构变化才需要重新 capture

strace 对比：

.. code:: text

   ; cudaGraphExecUpdate (参数更新)
   0x4e × 1     ← 更新图参数 (1 次 ioctl)
   ; 无 capture, 无 instantiate 开销
   
   ; cudaGraphInstantiate (重新实例化)
   0x4e × 2     ← capture (1) + instantiate (1)
   0x2a × 1     ← 额外分配

Updatable Graph 适用于**图结构固定、仅参数变化**的场景，如模型推理中
batch size 或输入指针变化。这是 CUDA Graph 实际部署中最重要的优化。

--------------

关键发现
-----------

1. **Graph capture 不产生额外 ioctl** — ``cudaStreamBeginCapture``
   只是切换 stream 状态为"记录模式"，后续 launch 的 ioctl 路径与
   普通 launch 完全相同。capture 的成本就是一次 kernel launch 的成本。

2. **Instantiate 是最重操作** — ``cudaGraphInstantiate`` 占重建时间
   的 78%。它内部包括：验证命令序列 → 构建 GPU 调度器可执行描述符
   → 分配内部结构（``ioctl(0x2a)``）。

3. **单 kernel 图收益有限** — 1.11× 加速比说明 launch 开销本身不是
   主要瓶颈。Graph 的真正应用场景是**多节点图**——vLLM、TensorRT-LLM
   等推理框架将一个模型的数十个算子捕获为一张图，此时 launch 次数
   减少 10-100 倍，加速效果显著。

4. **Updatable Graph 是必选项** — 对于实际部署，应尽可能使用
   ``cudaGraphExecUpdate`` 而非每次重新 instantiate。将静态结构
   （图拓扑）与动态参数（指针、大小）分离，是 CUDA Graph 性能优化
   的核心模式。

*分析基于 CUDA 13.1 / Driver 595.58.03 / RTX 4060 Laptop GPU。单 kernel 图
测试，实际多节点图的加速比可达 5-10×。*

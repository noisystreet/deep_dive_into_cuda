Green Context：GPU 资源分区与轻量级上下文
================================================

   Green Context 是 CUDA 12.x+ 引入的**硬件资源分区**抽象。与
   传统 Context 管理整个 GPU 不同，Green Context 允许将一个 GPU 的

.. admonition:: 你知道吗？

   Green Context 的"green"这个名字有一个有趣的来源：它不是指
   环保，而是指"**轻量级、可快速创建**"的上下文，就像
   "green thread"（用户态线程）对比 OS 线程一样。NVIDIA 的文档
   甚至明确指出 green context 不是线程安全的——它只能被一个线程
   使用，这进一步印证了与 green thread 的类比。在实际部署中，
   Green Context 最常见的用途是 MIG（Multi-Instance GPU）的软件
   替代方案——当 GPU 不支持 MIG 硬件分区时，Green Context 在驱动
   层提供类似的分区能力。

   SM（流多处理器）划分为多个 partition，每个 partition 创建独立的
   轻量级上下文。这是 MIG（Multi-Instance GPU）在单 GPU 内的软件级
   等价实现。

   环境: CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) / RTX 4060

   测试程序: ``examples/greenctx_demo.cu``

--------------

为什么需要 Green Context？
---------------------------

传统 CUDA Context 是整个 GPU 的抽象——无论应用只使用 1 个 kernel 还是
100 个，context 都会占用全部 SM 资源。Green Context 解决了三个问题：

1. **资源隔离** — 多个工作负载可以运行在独立的 SM partition 上，干扰
   最小化
2. **轻量级创建** — Green Context 比传统 CUDA context 创建快，不分配
   完整地址空间
3. **细粒度分区** — 可以精确控制一个应用使用多少 SM

API 流程
----------

创建 Green Context 的完整流程：

.. code:: text

   1. cuDevicePrimaryCtxRetain()      ← 保留 primary context
   2. cuDeviceGetDevResource(SM)      ← 查询设备 SM 总数和分区约束
   3. cuDevSmResourceSplitByCount()   ← 分割 SM 为 partition + remainder
   4. cuDevResourceGenerateDesc()     ← 从 resource 列表生成 descriptor
   5. cuGreenCtxCreate()              ← 创建 Green Context
   6. cuCtxFromGreenCtx()             ← 转为 CUcontext 供 API 使用
   7. cuCtxSetCurrent()               ← 设置为当前上下文
   8. cuMemAlloc / cuLaunchKernel     ← 正常 CUDA 操作

关键 API 说明：

- ``cuDevSmResourceSplitByCount`` 将 SM 资源分割为 partition 和
  remainder。分割粒度受 ``minSmPartitionSize`` 约束——RTX 4060 上
  最小 partition 为 2 个 SM（coscheduled alignment 也是 2）。
- ``cuGreenCtxCreate`` 的 flags 参数必须包含
  ``CU_GREEN_CTX_DEFAULT_STREAM``。
- Green Context 共享 primary context 的 module 命名空间——加载的
  cubin/PTX 可以从 green context 访问。
- 使用后需 ``cuCtxSetCurrent(primary_ctx)`` 恢复，再
  ``cuGreenCtxDestroy`` 销毁。

--------------

系统调用特征
---------------

与常规 context 创建相比，Green Context 的 ioctl 特征：

.. list-table:: ioctl 对比 (greenctx_demo)
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - ioctl 号
     - 常规 ctx
     - Green ctx
     - 新增
     - 说明
   * - 总数
     - ~400
     - 1161
     - **2.9×**
     - 含 nvcc 子进程
   * - ``0x49``
     - 24
     - 79
     - **3.3×**
     - SM partition 操作
   * - ``0x4f``
     - 0
     - 69
     - **新增**
     - Green ctx 特有
   * - ``0x2a`` (alloc)
     - 25
     - 367
     - **14×**
     - SM partition 分配
   * - ``0x2b`` (fence)
     - 25
     - 282
     - **11×**
     - partition fence
   * - ``0x4e`` (launch)
     - 25
     - 71
     - **2.8×**
     - 含 nvcc 子进程

``0x4f`` 是 Green Context 场景下特有的 ioctl（在常规 context 测试中
未出现），推测对应 **SM partition 的硬件资源提交操作**。

--------------

实验结果
---------

.. list-table:: Green Context 测试结果 (RTX 4060, 24 SM)
   :header-rows: 1
   :widths: 35 25 40

   * - 操作
     - 结果
     - 说明
   * - SM 总数
     - 24
     - RTX 4060 有 24 个 SM
   * - 最小 partition
     - 2 SM
     - ``minSmPartitionSize`` = 2
   * - 分割后剩余
     - 22 SM
     - 剩余 SM 仍归 primary ctx
   * - Green ctx 创建
     - 成功
     - 2 SM partition
   * - 分配显存
     - 成功
     - 在 green ctx 地址空间
   * - Kernel launch
     - 成功
     - 运行在 2 SM partition 上
   * - 结果验证
     - 正确
     - Result[0]=0, Result[1]=1

关键发现
-----------

1. **Green Context 是硬件支持的 SM 分区** — 不同于纯软件模拟的 context
   切换，Green Context 在 GPU 硬件层面创建独立的 SM partition。
   ``0x4f`` ioctl 的出现表明驱动向 GPU 提交了分区配置。

2. **分区粒度由硬件决定** — RTX 4060 的最小 SM partition 为 2（与
   coscheduling alignment 一致）。这是 Ada Lovelace 架构的硬件约束，
   与 MIG 的粒度（Ampere 上为 1/7 个 GPU）不同。

3. **Green Context 共享主 context 的 module** — ``cuModuleLoadData``
   加载的 cubin 在 primary context 上注册后，green context 可以
   通过 ``cuModuleGetFunction`` 获取 kernel 入口。这避免了在每个
   green context 中重复加载。

4. **cuGreenCtxStreamCreate 不可用** — 在 RTX 4060 (sm_89) 上，
   ``cuGreenCtxStreamCreate`` 返回 ``CUDA_ERROR_INVALID_ARGUMENT``。
   使用 ``cuStreamCreate`` + green context 设为 current 即可。
   这可能与 Ada Lovelace 的 workqueue 实现有关。

5. **比创建完整 context 轻量** — Green Context 不创建独立 GPU 虚拟
   地址空间（共享 primary context 的空间），因此 ``cuMemAlloc`` 和
   kernel launch 的 ioctl 开销与常规 context 一致，但创建阶段的
   ``0x49`` 和 ``0x2a`` 调用显著减少。

6. **适用场景** — Green Context 适合多工作负载隔离、实时性要求高
   （避免 kernel 间的 SM 抢占）和 GPU 分区虚拟化场景。对于常规
   单应用，使用 ``cuCtxCreate`` + stream 即可，不需要 Green Context。

*分析基于 CUDA 13.1 / Driver 595.58.03 / RTX 4060 Laptop GPU (24 SM, sm_89).*
*Green Context 需要 CUDA 12.0+ 和 sm_86+ GPU.*

# Deep Dive Into CUDA

从 ``vector_add.cu`` 一个简单的向量加法程序出发，贯穿 **源码 → 编译 → 二进制 → 加载 → 驱动 → GPU 执行** 的完整链路，深入分析 CUDA 编译器工具链与运行时驱动的内部架构。

## 文档

在线文档（Read the Docs）：

> TODO: 部署后添加链接

## 目录结构

```
deep_dive_into_cuda/
├── source/                        # Sphinx 文档源
│   ├── preface/                   # 前言
│   ├── chapter_01_compilation/    # 编译过程深度分析
│   ├── chapter_02_toolchain/      # nvcc 工具链逆向分析
│   ├── chapter_03_runtime/        # 运行时与驱动
│   ├── appendix/                  # 附录
│   └── _static/                   # Mermaid 图与自定义 CSS
├── examples/                      # 示例源码
│   ├── vector_add.cu
│   ├── CMakeLists.txt
│   └── demo_build.sh
├── scripts/                       # pre-commit 检查脚本
├── Makefile                       # 构建入口
└── _build/html/                   # 构建产物（make html 后生成）
```

## 内容覆盖

| 章节 | 内容 |
|---|---|
| **编译过程** | NVCC 编译流水线、中间产物分析、Fat Binary 结构 |
| **工具链逆向** | nvcc / cudafe++ / cicc / ptxas / nvlink 逆向分析 |
| **运行时与驱动** | libcudart 分析、GPU 驱动接口 strace 分析 |

## 本地构建

```bash
git clone <repo>
cd deep_dive_into_cuda
git config --local core.hooksPath .githooks   # 启用 pre-commit RST 检查
pip install -r requirements.txt
make html       # 构建 HTML（CI 使用 make html SPHINXOPTS="-W"）
make serve      # 构建并在 localhost:8000 启动预览
bash scripts/precommit-check.sh             # 手动运行 RST 语法检查
```

## 许可证

MIT

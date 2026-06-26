# agents.md — Deep Dive Into CUDA 项目

## 项目概述

本项目编写一本 **Deep Dive Into CUDA** 技术文档，使用 reStructuredText（`.rst`）格式，基于 Sphinx 构建。

- 文档源目录：`source/`
- 构建输出：`_build/html/`（`make html` 后生成）
- 目标读者：有 CUDA 编程基础、希望深入理解 CUDA 编译器工具链与运行时驱动的开发者
- 分析对象：基于 `vector_add.cu` 一个简单的向量加法程序，贯穿 **源码 → 编译 → 二进制 → 加载 → 驱动 → GPU 执行** 的完整链路
- 环境：CUDA 13.1 / Driver 595.58.03 / sm_89 (Ada Lovelace) / Linux x86-64
- 平台：Linux x86-64

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `source/preface/index.rst` | 前言：全景架构总览 |
| `source/index.rst` | Sphinx 根文档（toctree 入口） |
| `source/chapter_01_compilation/` | 编译过程深度分析（3 篇：编译流水线、中间产物、Fat Binary 结构） |
| `source/chapter_02_toolchain/` | nvcc 工具链分析（5 篇：nvcc / cudafe++ / cicc / ptxas / nvlink） |
| `source/chapter_03_runtime/` | 运行时与驱动（2 篇：GPU 驱动接口、libcudart 分析） |
| `source/appendix/` | 附录（资源推荐、术语表） |
| `source/conf.py` | Sphinx 构建配置（sphinx_rtd_theme） |
| `source/_static/*.mmd` | Mermaid 示意图（全景图、流水线、Fat Binary 层级、IOCTL 饼图、调用链） |
| `source/_static/custom.css` | 自定义 CSS |
| `Makefile` | 构建入口（`make html` / `make clean` / `make serve`） |
| `examples/` | 可运行的示例代码（vector_add.cu + CMakeLists.txt + demo_build.sh） |
| `scripts/precommit-check.sh` | 预提交检查脚本（验证 RST 文档语法） |
| `requirements.txt` | 构建依赖（sphinx, sphinx-rtd-theme, sphinxcontrib-mermaid） |
| `.readthedocs.yaml` | Read the Docs 构建配置 |
| `LICENSE` | MIT 许可证 |
| `.gitignore` | 版本控制忽略规则 |
| `agents.md` | **本文件**：AI 助手的工作上下文和约束 |

## 通用约束

1. **许可证**：本项目采用 MIT 许可证，详见 `LICENSE` 文件
2. **文档格式**：使用 reStructuredText（`.rst`）格式，中文写作
3. **git hooks**：clone 后首次提交前，运行以下命令启用 pre-commit 检查：

   ```bash
   git config --local core.hooksPath .githooks
   ```

   否则 pre-commit 检查不会自动生效。
4. **引用源码**：使用仓库内相对路径（如 ``examples/vector_add.cu``）或 GitHub blob 链接；不要使用 ``file:///`` 协议
5. **避免冗余**：不创建不必要的文件，优先编辑已有文件
6. **权限**：不做 `git push --force`、`reset --hard` 等破坏性操作
7. **代码示例**：在文档中引用代码时，说明其所属文件和行号范围
8. **示例验证**：所有 `.cu` 示例代码应保证可编译运行

## 文档写作规范

### 文档结构
- 每篇文档应有标题
- 按章节组织，章节层级不超过三级
- 内容末尾标注生成日期和项目名称

### 引用规范
- 引用源码文件使用仓库相对路径（``examples/...``）或 GitHub 链接
- 引用 API 或概念使用 `` ` `` 反引号标记
- 关键代码片段应提供文件定位

### 内容深度
- 概念讲解与实际分析结果相结合
- 复杂流程配合 Mermaid 图表说明
- 关键工具用表格列出其大小、依赖、输入输出
- 避免大段堆叠代码，优先提炼核心模式

### 写作风格（核心：层次递进，证据驱动）

**禁止罗列结论**。每一个知识点都必须有推导过程，遵循"是什么 → 为什么 → 怎么分析 → 发现了什么"的递进链条。

- **层次递进**：从直观的编译命令出发，逐步深入到二进制分析、逆向分析、运行时追踪。每一节都遵循：表象现象 → 可能的机制 → 通过工具验证 → 得出结论。
- **证据驱动**：每一个论断必须有工具输出或源码引用佐证（strace 日志、strings 输出、nm 符号表、PTX/SASS 反汇编）。没有证据支撑的观点都是空谈。
- **避免知识点罗列**：每个新概念必须有上下文铺垫才引入。分析工具的用法（strace、strings、nm、cuobjdump、nvdisasm）应在具体分析场景中自然引出。
- **工具即视角**：每一章对应一类分析工具/视角——编译日志分析、中间文件分析、逆向分析、运行时追踪。每一篇文档应让读者学会一种新的"看问题的方式"。
- **过渡自然**：段落之间、章节之间要有承上启下的过渡句。比如"上一节我们看到了编译器如何生成 PTX，接下来我们看看 ptxas 如何将 PTX 翻译为 GPU 能执行的机器码"。

## 写作路线图

已完成全部 11 篇分析文档的编写：

1. **前言** — 全景架构总览（Mermaid 图：5 层全链路）
2. **第 1 章：编译过程深度分析**
   - 1.1 编译流水线详解（Mermaid 图：双路径 11 步）
   - 1.2 编译中间产物分析（源码 489B → 预处理 1.3MB 的 2700 倍膨胀）
   - 1.3 Fat Binary 结构分析（Mermaid 图：容器→ELF→SASS）
3. **第 2 章：nvcc 工具链分析**
   - 2.1 nvcc 分析（Driver Compiler，33 次 execve）
   - 2.2 cudafe++ 分析（基于 EDG C++ 6.7）
   - 2.3 cicc 分析（基于 LLVM 7.0.1 / NVVM，77 MB）
   - 2.4 ptxas 分析（两阶段架构：Backend + Finalizer）
   - 2.5 nvlink 分析（dlopen 加载 libnvvm / libtileiras）
4. **第 3 章：运行时与驱动**
   - 3.1 GPU 驱动接口分析（strace 追踪，Mermaid 图：IOCTL 分布 + 调用链）
   - 3.2 libcudart 分析（429 个 API，薄包装层）

## 构建方法

```bash
# 安装依赖
pip install -r requirements.txt

# 构建 HTML 文档
make html

# 构建产物位于 _build/html/
# 本地预览
make serve
```

自动部署到 Read the Docs 后，文档会自动构建并托管。本地构建也可通过 `make html` 完成。

## Cursor Cloud specific instructions

本项目是一个 **Sphinx 文档站点**（中文 CUDA 分析教程），"运行应用"即构建并预览 HTML 文档。依赖已由启动 update script (`pip install -r requirements.txt`) 安装好（`sphinx` / `sphinx-rtd-theme` / `sphinxcontrib-mermaid`）。常用命令见 `README.md` 与 `Makefile`，下面只记录非显而易见的注意事项：

- **构建**：`make html`（产物在 `_build/html/`）。本地开发用 `make html` 即可；不要加 `-W`。CI（`.github/workflows/ci.yml`）使用 `make html SPHINXOPTS="-W"` 把警告当错误。
- **预览**：`make serve` 会先构建再用 `python3 -m http.server` 启动预览，默认端口 8000。也可直接 `cd _build/html && python3 -m http.server 8000`。
- **Lint / RST 检查**：`bash scripts/precommit-check.sh`。脚本会运行 Sphinx 语法解析；内联标记风格仅作提示，不阻塞 CI。
- **git hooks**（仅在需要提交触发 pre-commit RST 检查时）：`git config --local core.hooksPath .githooks`。
- 修改 `.rst` 内容后无热重载，需重新 `make html` 才能在预览中看到更新。
- **Mermaid 图**：存放在 `source/_static/*.mmd`，通过 `.. mermaid:: ../_static/xxx.mmd` 在 RST 中引用。修改 `.mmd` 文件后需要重新 `make html`。
- **分析工具**：项目中使用的分析工具（strace、strings、nm、file、ldd、readelf、cuobjdump、nvdisasm、ptxas 等）是 Linux 标准工具或 CUDA 工具包自带，不依赖 Python 包。

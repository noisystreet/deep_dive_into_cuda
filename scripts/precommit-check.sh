#!/bin/bash
# RST 文档预提交检查脚本
set -e

if ! python3 -c "import sphinx, sphinxcontrib.mermaid" 2>/dev/null; then
    echo "错误: 未安装 Sphinx 依赖。请先运行: pip install -r requirements.txt"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$1" = "--hook" ]; then
    FILES=$(git --git-dir="$PROJECT_ROOT/.git" diff --cached --name-only --diff-filter=ACM | grep '\.rst$' || true)
elif [ "$1" = "--staged" ]; then
    FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.rst$' || true)
else
    FILES=$(find "$PROJECT_ROOT/source" -name '*.rst' | sort)
fi

if [ -z "$FILES" ]; then
    echo "没有 RST 文件需要检查。"
    exit 0
fi

echo "检查 RST 文件语法..."

BUILD_DIR="$PROJECT_ROOT/_build/precommit-check"
rm -rf "$BUILD_DIR"

if python3 -m sphinx -b dummy "$PROJECT_ROOT/source" "$BUILD_DIR" 2>/tmp/sphinx_err.txt 1>/dev/null; then
    if grep -qE '(WARNING|ERROR)' /tmp/sphinx_err.txt 2>/dev/null; then
        echo "⚠ 构建成功，但有警告:"
        grep -E '(WARNING|ERROR)' /tmp/sphinx_err.txt
        exit 2
    fi
    echo "✓ 所有 RST 文件语法正确。"
else
    echo "✗ RST 语法错误！"
    cat /tmp/sphinx_err.txt
    exit 1
fi

rm -rf "$BUILD_DIR"

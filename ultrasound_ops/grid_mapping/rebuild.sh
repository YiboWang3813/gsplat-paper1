#!/usr/bin/env bash
# 遇到错误立即停止
set -e

# --- 自动检测包名 ---
# 逻辑：寻找当前目录下包含 __init__.py 的第一个文件夹名
PKG=$(find . -maxdepth 2 -name "__init__.py" -print -quit | cut -d'/' -f2)

if [ -z "$PKG" ]; then
    echo "错误: 未能在当前目录找到 Python 包 (缺少 __init__.py)"
    exit 1
fi

echo "=================================================="
echo "  OmniUltra 模块重构工具: [ $PKG ]"
echo "=================================================="

echo "==> 1. 清理旧的编译中间产物..."
rm -rf build dist *.egg-info
# 删除当前目录及包目录下的 .so 文件
find . -name "*.so" -type f -delete

echo "==> 2. 原地编译 CUDA 扩展 (build_ext --inplace)..."
python setup.py build_ext --inplace

echo "==> 3. 以可编辑模式安装到环境 (pip install -e .)..."
# 使用 --no-deps 避免重复检查依赖，加快安装速度
python -m pip install --no-deps -e .

echo "=================================================="
echo "  ✅ $PKG 重新编译并安装完成！"
echo "=================================================="
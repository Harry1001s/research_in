#!/usr/bin/env bash
# 新项目环境初始化。可重复运行。
set -uo pipefail
cd "$(dirname "$0")/.."

ok()   { echo "✅ $*"; }
warn() { echo "⚠️  $*"; }

echo "== 工具检查 =="
command -v git    >/dev/null && ok "git"    || warn "缺少 git"
command -v uv     >/dev/null && ok "uv"     || warn "缺少 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"
if command -v nvidia-smi >/dev/null; then
  ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';' -)"
else
  warn "未检测到 NVIDIA GPU（CPU / Mac 可以忽略）"
fi

echo "== Python 依赖 =="
if command -v uv >/dev/null; then
  uv sync && ok "uv sync 完成"
  echo "   需要 PyTorch / W&B 时：uv sync --extra torch --extra track"
fi

echo "== 冒烟测试 =="
command -v uv >/dev/null && uv run pytest -q && ok "测试通过"

chmod +x scripts/*.sh
echo "== 完成 =="

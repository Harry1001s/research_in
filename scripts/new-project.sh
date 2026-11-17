#!/usr/bin/env bash
# 从模板创建新研究项目：bash scripts/new-project.sh <项目名> [--public]
# 需要 gh 已登录（gh auth login）。
set -euo pipefail
NAME="${1:?用法: new-project.sh <项目名> [--public]}"
VIS="${2:---private}"
OWNER="$(gh api user -q .login)"
TEMPLATE="${TEMPLATE:-$OWNER/research-init}"

gh repo create "$NAME" --template "$TEMPLATE" "$VIS" --clone
cd "$NAME"
bash scripts/bootstrap.sh
echo
echo "项目已创建：$(pwd)"

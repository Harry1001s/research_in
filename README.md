# research-init：ML 研究项目模板

每个新研究项目都从这个模板生成。

```
scripts/             ← bootstrap.sh / new-project.sh
src/project/train.py ← 配置驱动的训练入口，自动记录 run（占位模型，替换即可）
configs/  tests/  data/  experiments/  notebooks/
```

## 一次性设置

1. **把本模板推到 GitHub，并设为 Template**
   ```bash
   cd research-init
   git add -A && git commit -m "init template"
   gh repo create research-init --private --source=. --push
   gh repo edit --template
   ```
2. **安装工具**：`uv`、`gh`。

## 每个新项目

```bash
bash research-init/scripts/new-project.sh my-new-idea   # 从模板创建、clone、bootstrap
```

## 替换成你的模型

只需改 `src/project/train.py` 中的 `build_data()` 和 `train()`。run 目录、`meta.json`（commit/seed/环境）和 `metrics.jsonl` 的记录逻辑保持不动。需要 PyTorch 或 W&B 时运行 `uv sync --extra torch --extra track`。

"""训练入口：配置驱动，自动记录 run 元信息。

用法:
    uv run python -m project.train --config configs/default.yaml [--seed 0] [--debug]

模板里的模型是一个 numpy 写的逻辑回归占位，用来验证整套流程能跑通。
替换成你的真实模型时，只需要改 `build_data` 和 `train` 两个函数，
run 目录、元信息和指标记录的逻辑保持不动。
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from project.utils import load_config, set_seed


def git_info() -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unknown"

    return {
        "commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "dirty": run(["git", "status", "--porcelain"]) != "",
    }


def make_run_dir(cfg: dict[str, Any], seed: int, debug: bool) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = cfg.get("name", "run")
    run_id = f"{stamp}_{name}_s{seed}" + ("_debug" if debug else "")
    run_dir = Path(cfg.get("output_dir", "experiments")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_data(cfg: dict[str, Any], rng: np.random.Generator, debug: bool):
    """占位数据：线性可分的二分类。替换成真实的数据加载。"""
    n = cfg["data"]["n_samples"] // (20 if debug else 1)
    d = cfg["data"]["n_features"]
    w_true = rng.normal(size=d)
    X = rng.normal(size=(n, d))  # (N, D)
    y = (X @ w_true + 0.5 * rng.normal(size=n) > 0).astype(np.float64)  # (N,)
    split = int(n * (1 - cfg["data"]["val_frac"]))
    return (X[:split], y[:split]), (X[split:], y[split:])


def train(cfg: dict[str, Any], train_set, val_set, log, debug: bool) -> dict[str, float]:
    """占位训练：逻辑回归 + 梯度下降。替换成真实模型。"""
    X, y = train_set
    Xv, yv = val_set
    w = np.zeros(X.shape[1])
    lr = cfg["train"]["lr"]
    steps = cfg["train"]["steps"] // (20 if debug else 1)
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))

    for step in range(1, steps + 1):
        p = sigmoid(X @ w)
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        w -= lr * X.T @ (p - y) / len(y)
        if step % cfg["train"]["log_every"] == 0 or step == steps:
            val_acc = float(np.mean((sigmoid(Xv @ w) > 0.5) == yv))
            log({"step": step, "train_loss": float(loss), "val_acc": val_acc})

    return {"val_acc": float(np.mean((sigmoid(Xv @ w) > 0.5) == yv))}


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 config 中的 seed")
    ap.add_argument("--debug", action="store_true", help="小数据、少步数，快速跑通")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    rng = set_seed(seed)
    run_dir = make_run_dir(cfg, seed, args.debug)

    (run_dir / "config.yaml").write_text(Path(args.config).read_text())
    meta = {
        "config_path": args.config,
        "seed": seed,
        "debug": args.debug,
        "python": platform.python_version(),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        **git_info(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    metrics_f = open(run_dir / "metrics.jsonl", "a")
    log_f = open(run_dir / "train.log", "a")

    def log(record: dict[str, Any]) -> None:
        metrics_f.write(json.dumps(record) + "\n")
        metrics_f.flush()
        line = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    train_set, val_set = build_data(cfg, rng, args.debug)
    final = train(cfg, train_set, val_set, log, args.debug)
    log({"final": True, **final})

    metrics_f.close()
    log_f.close()
    print(f"run_dir={run_dir}")
    return run_dir


if __name__ == "__main__":
    main()

import json

import yaml

from project.train import main


def test_debug_run_end_to_end(tmp_path):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["output_dir"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = main(["--config", str(cfg_path), "--debug", "--seed", "1"])

    for f in ["config.yaml", "meta.json", "metrics.jsonl", "train.log"]:
        assert (run_dir / f).exists(), f
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["seed"] == 1 and meta["debug"] is True
    final = json.loads((run_dir / "metrics.jsonl").read_text().strip().splitlines()[-1])
    assert final["final"] and 0.0 <= final["val_acc"] <= 1.0


def test_seed_reproducible(tmp_path):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["output_dir"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    def final_acc():
        d = main(["--config", str(cfg_path), "--debug", "--seed", "3"])
        return json.loads((d / "metrics.jsonl").read_text().strip().splitlines()[-1])["val_acc"]

    import time
    a = final_acc()
    time.sleep(1.1)  # run_id 以秒为粒度
    assert a == final_acc()

"""V.1 step 1: train the broken model (ERM) with the DFR paper's Waterbirds recipe (Appendix C).

ResNet-50 (torchvision IMAGENET1K_V1), SGD momentum 0.9, constant lr 1e-3, weight decay 1e-3, batch 32,
100 epochs, RandomResizedCrop + flip, no early stopping (the final epoch is the model). fp32.
Validation is scored every 10 epochs for the log only; it chooses nothing.

    ~/vision_env/.venv/bin/python -m exp_v.v1_erm --out artifacts/v/broken
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from exp_v import wb


def evaluate(model, split):
    _, logits, y, g = wb.features_and_logits(model, split)
    return wb.score(logits.argmax(1), y, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/v/broken")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)

    model = wb.resnet50().cuda()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    dl = wb.loader("train", train=True)
    log = open(out / "train_log.jsonl", "w")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for x, y, _ in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            out_ = model(x)
            loss = torch.nn.functional.cross_entropy(out_, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); correct += (out_.argmax(1) == y).sum().item(); n += len(y)
        rec = {"epoch": ep, "loss": round(tot / n, 5), "train_acc": round(100 * correct / n, 2), "min": round((time.time() - t0) / 60, 1)}
        if ep % 10 == 0 or ep == a.epochs:
            v = evaluate(model, "val")
            rec.update(val_worst=v["worst"], val_mean_weighted=v["mean_weighted"])
            torch.save(model.state_dict(), out / "model.pt")
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

    torch.save(model.state_dict(), out / "model.pt")
    res = {"recipe": vars(a) | {"lr": 1e-3, "wd": 1e-3, "momentum": 0.9, "batch": 32, "precision": "fp32",
                                "backbone": "torchvision ResNet50 IMAGENET1K_V1"},
           "val": evaluate(model, "val"), "test": evaluate(model, "test"), "minutes": round((time.time() - t0) / 60, 1)}
    (out / "erm_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

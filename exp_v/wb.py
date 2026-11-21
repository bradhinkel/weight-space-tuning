"""Waterbirds loading and scoring for Experiment V (scoring side; never imported in the sandbox).

Groups g = 2*y + place: 0 landbird/land, 1 landbird/water, 2 waterbird/land, 3 waterbird/water.
Mean accuracy is the published definition (DFR Table 2, after Sagawa et al.): group accuracies
weighted by each group's share of the TRAINING split. The plain average is reported as well.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

ROOT = Path.home() / "datasets" / "waterbirds"
RAW = ROOT / "extracted"
MANIFESTS = ROOT / "manifests"
GROUPS = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# DFR Appendix C; eval = resize to 256x256 then center-crop 224 (the DFR code's Waterbirds eval transform)
TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 4 / 3), interpolation=T.InterpolationMode.BILINEAR),
                      T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
EVAL_TF = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def manifest(split):
    return pd.read_csv(MANIFESTS / f"{split}.csv")


def train_group_weights():
    g = manifest("train").g.value_counts().sort_index()
    return (g / g.sum()).to_numpy()


class WB(torch.utils.data.Dataset):
    def __init__(self, split, tf):
        self.m, self.tf = manifest(split), tf

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        r = self.m.iloc[i]
        return self.tf(Image.open(RAW / r.img_filename).convert("RGB")), int(r.y), int(r.g)


def loader(split, train=False, bs=32, workers=12):
    ds = WB(split, TRAIN_TF if train else EVAL_TF)
    return torch.utils.data.DataLoader(ds, batch_size=bs if train else 128, shuffle=train, drop_last=False,
                                       num_workers=workers, pin_memory=True, persistent_workers=train)


def resnet50(pretrained=True):
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet50(weights=w)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m


def score(pred, y, g):
    """pred, y, g: integer arrays. Returns per-group accuracy, worst-group, weighted mean, plain mean."""
    pred, y, g = map(np.asarray, (pred, y, g))
    acc = np.array([(pred[g == k] == y[g == k]).mean() for k in range(4)])
    return {"group_acc": {GROUPS[k]: round(100 * acc[k], 2) for k in range(4)},
            "worst": round(100 * acc.min(), 2), "worst_group": GROUPS[int(acc.argmin())],
            "mean_weighted": round(100 * float(acc @ train_group_weights()), 2),
            "mean_plain": round(100 * float((pred == y).mean()), 2), "n": int(len(y))}


@torch.no_grad()
def features_and_logits(model, split, device="cuda"):
    """Penultimate features (input to fc), logits, labels, groups for a split, in manifest order."""
    model.eval().to(device)
    feats, logits, ys, gs = [], [], [], []
    hook = model.fc.register_forward_hook(lambda mod, inp, out: feats.append(inp[0].float().cpu()))
    for x, y, g in loader(split):
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        ys.append(y), gs.append(g)
    hook.remove()
    return torch.cat(feats).numpy(), torch.cat(logits).numpy(), torch.cat(ys).numpy(), torch.cat(gs).numpy()

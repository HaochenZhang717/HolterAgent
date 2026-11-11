"""B3: an image CNN on the rendered ECG images the agent sees, in the style of Xiao et al. 2018.

Source: Xiao R, Xu Y, Pelter MM, Mortara DW, Hu X. A Deep Learning Approach to Examine Ischemic ST Changes in
Ambulatory ECG Recordings. AMIA Jt Summits Transl Sci Proc 2018;2017:256-262 (PMC5961830). They rendered 10-s
single-lead LTSTDB strips on standard ECG paper (0.04 s x 0.1 mV grid), kept the feature layers of Inception-V3
pretrained on ImageNet and trained new final layers (50 epochs), balancing cases and controls; the threshold was
chosen per test session (Youden index).

Here (baselines/learned_common.py protocol): one model per image kind of outputs/baselines/_bundle (strip, overlay,
trend, beats; strip_pair = strip_ref + strip through a shared backbone with concatenated features). Differences
from Xiao et al.: whole multi-lead images (as the agent sees them), positives = ischemic only, the threshold comes
from the development predictions (common.evaluate), and only pixel shifts as augmentation (one rendered image per
candidate, so no temporal oversampling).

- layout: no rescaling; each image is centred on a white canvas of its view's largest size (1360x370 strips,
  1200x425 trend, 605x350 beats/overlay), so px per mV and px per s are the same for every candidate; Inception-V3
  is fully convolutional up to global average pooling, so it takes these sizes as they are;
- augmentation (training only): random shift of up to +-16 px horizontally and +-8 px vertically; no flips;
- model: torchvision inception_v3 (IMAGENET1K_V1) without the auxiliary classifier; head dropout 0.5 + linear;
- training: BCE with pos_weight = negatives / positives of the training folds, AdamW (weight decay 1e-4), batch 16,
  one epoch of linear warm-up then a constant learning rate, bf16 autocast;
- grid (record-grouped 5-fold CV on development, pooled out-of-fold AUC): frozen backbone + head (Xiao et al.)
  with lr 1e-3 / 1e-2 at 5-50 epochs, and full fine-tuning with lr 3e-5 / 1e-4 at 2-30 epochs; the number of epochs
  is one global choice per configuration, never early stopping on a held-out fold;
- final: the chosen configuration retrained on each training split; development scores are the held-out fold
  predictions, the test score is the mean of the 5 fold models (test images are loaded only here);
- fusion (fixed before training): the unweighted mean of the strip, overlay, trend and beats probabilities.

    python -m baselines.image_cnn grid --view strip       # GPU: development grid
    python -m baselines.image_cnn select                  # grid table and the chosen configuration per view
    python -m baselines.image_cnn final --view strip      # GPU: chosen configuration per fold + test scores
    python -m baselines.image_cnn finish                  # rows imagecnn_<view> and imagecnn_fusion

Intermediate files: outputs/baselines/_image_cnn/{grid,final}/<view>/..., selection.json, grid.md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np

from baselines.common import OUT, auc
from baselines.learned_common import Bundle, dev_folds, finish, load_bundle

WORK = Path(os.environ.get("IMAGE_CNN_OUT", OUT / "_image_cnn"))
VIEWS: dict[str, tuple[str, ...]] = {"strip": ("strip",), "overlay": ("overlay",), "trend": ("trend",),
                                     "beats": ("beats",), "strip_pair": ("strip_ref", "strip")}
FUSED = ("strip", "overlay", "trend", "beats")   # views averaged by imagecnn_fusion
MAIN_CANDIDATES = FUSED + ("fusion",)            # rows eligible as the main row (strip_pair is supplementary)
CANVAS = {"strip": (370, 1360), "strip_ref": (370, 1360), "trend": (425, 1200), "beats": (350, 605),
          "overlay": (350, 605)}                 # (height, width): the largest native size of each kind
SHIFT = (8, 16)                                  # max augmentation shift (vertical, horizontal) in px
GRID: list[tuple[str, float]] = [("frozen", 1e-3), ("frozen", 1e-2), ("full", 3e-5), ("full", 1e-4)]
EPOCHS = {"frozen": (5, 10, 20, 30, 50), "full": (2, 5, 10, 20, 30)}
BATCH, WEIGHT_DECAY, DROPOUT, N_FOLDS = 16, 1e-4, 0.5, 5
# CPU fallback (external evaluation, 2026-10-23: GPU server GPU pods were preempted within minutes): on a GPU everything is
# as before (bf16 autocast); on a CPU the same training runs in float32.


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast():
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if _device() == "cuda" else contextlib.nullcontext()
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet; torchvision maps them to Inception's input


def config_name(mode: str, lr: float) -> str:
    return f"{mode}_lr{lr:g}"


def letterbox(path: Path, kind: str) -> np.ndarray:
    """The image as RGB uint8 [3, H, W], centred on its view's white canvas without rescaling. A candidate whose image
    could not be rendered (path None, e.g. too few clean beats; one test case of the third test set) gets the blank
    canvas."""
    from PIL import Image
    h, w = CANVAS[kind]
    if path is None:
        return np.asarray(Image.new("RGB", (w, h), "white")).transpose(2, 0, 1).copy()
    im = Image.open(path).convert("RGB")
    assert im.height <= h and im.width <= w, (path, im.size)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.asarray(canvas).transpose(2, 0, 1).copy()


def load_images(b: Bundle, idx: np.ndarray, view: str, device):
    """uint8 tensors [n, 3, H, W] on `device`, one per image kind of the view."""
    import torch
    return [torch.from_numpy(np.stack([letterbox(b.image(b.ids[i], kind), kind) for i in idx])).to(device)
            for kind in VIEWS[view]]


def build_net(n_images: int):
    """Inception-V3 backbone shared by the view's images, features concatenated, dropout + linear head."""
    import torch
    import torch.nn as nn
    from torchvision.models import Inception_V3_Weights, inception_v3

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)  # loads with the auxiliary classifier
            m.aux_logits, m.AuxLogits = False, None
            m.dropout, m.fc = nn.Identity(), nn.Identity()  # 2048-d pooled features
            self.backbone = m
            self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(2048 * n_images, 1))

        def features(self, xs):
            return torch.cat([self.backbone(x) for x in xs], 1)

        def forward(self, xs):
            return self.head(self.features(xs)).squeeze(1)

    return Net()


def _prepare(x, train: bool, gen):
    """uint8 [B, 3, H, W] -> normalised float, randomly shifted when training (white fill)."""
    import torch
    import torch.nn.functional as F
    x = x.float().div_(255)
    if train:
        dy, dx = SHIFT
        h, w = x.shape[-2:]
        xp = F.pad(x, (dx, dx, dy, dy), value=1.0)
        oy = torch.randint(0, 2 * dy + 1, (len(x),), generator=gen).tolist()
        ox = torch.randint(0, 2 * dx + 1, (len(x),), generator=gen).tolist()
        x = torch.stack([xp[i, :, a:a + h, c:c + w] for i, (a, c) in enumerate(zip(oy, ox))])
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def predict(net, imgs) -> np.ndarray:
    import torch
    net.eval()
    out = []
    with torch.no_grad(), _autocast():
        for s in range(0, len(imgs[0]), 32):
            out.append(torch.sigmoid(net([_prepare(x[s:s + 32], False, None) for x in imgs]).float()).cpu())
    return torch.cat(out).numpy()


def train(view: str, mode: str, lr: float, n_epochs: int, seed: int, imgs_tr, y_tr: np.ndarray,
          eval_sets: dict[str, list], eval_at: set[int], log=print) -> tuple[dict[str, np.ndarray], list[float]]:
    """Train one model; return {set: [len(eval_at), n] probabilities at the epochs in eval_at (ascending)}
    and the mean training loss of every epoch."""
    import torch
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = imgs_tr[0].device
    net = build_net(len(VIEWS[view])).to(dev).to(memory_format=torch.channels_last)
    frozen = mode == "frozen"
    if frozen:
        for p in net.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    y = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    pos_weight = torch.tensor((~y_tr).sum() / max(1, y_tr.sum()), device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = len(y_tr)
    steps = n // BATCH  # drop the incomplete last batch (reshuffled every epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / steps))
    preds: dict[str, list] = {k: [] for k in eval_sets}
    losses = []
    for ep in range(1, n_epochs + 1):
        net.train()
        if frozen:
            net.backbone.eval()  # fixed BN statistics and no dropout inside the frozen backbone
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for s in range(steps):
            bi = perm[s * BATCH:(s + 1) * BATCH].to(dev)
            xs = [_prepare(x[bi], True, gen) for x in imgs_tr]
            with _autocast():
                if frozen:
                    with torch.no_grad():
                        f = net.features(xs)
                    logit = net.head(f.float()).squeeze(1)
                else:
                    logit = net(xs)
                loss = lossf(logit.float(), y[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach().item()
        losses.append(tot / steps)
        if ep in eval_at:
            for k, imgs in eval_sets.items():
                preds[k].append(predict(net, imgs))
        log(f"  epoch {ep}/{n_epochs} loss {losses[-1]:.4f}")
    return {k: np.stack(v) for k, v in preds.items()}, losses


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)  # atomic, so a preempted job leaves no half-written fold


def grid(view: str, smoke: bool = False) -> None:
    """Development grid: every configuration on every fold, held-out predictions after every epoch.
    Only development images are loaded. Resumable (finished folds are skipped)."""
    import torch
    torch.backends.cudnn.benchmark = True
    b = load_bundle(signals=False)
    dev = b.dev
    fold, y = dev_folds(b), b.y[dev]
    imgs = load_images(b, dev, view, _device())
    root = WORK / ("smoke" if smoke else "grid") / view
    for mode, lr in GRID[:1] if smoke else GRID:
        n_ep = 2 if smoke else max(EPOCHS[mode])
        for k in range(1 if smoke else N_FOLDS):
            path = root / config_name(mode, lr) / f"fold{k}.npz"
            if path.exists():
                continue
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            t0 = time.time()
            print(f"{view} {config_name(mode, lr)} fold {k}: train {len(tr)}, held out {len(te)}", flush=True)
            p, losses = train(view, mode, lr, n_ep, seed=k, imgs_tr=[x[tr] for x in imgs], y_tr=y[tr],
                              eval_sets={"held": [x[te] for x in imgs]}, eval_at=set(range(1, n_ep + 1)),
                              log=lambda s: print(s, flush=True))
            sec = time.time() - t0
            held_auc = [auc(q, y[te]) for q in p["held"]]
            print(f"{view} {config_name(mode, lr)} fold {k}: {sec:.0f} s, held-out AUC by epoch "
                  f"{np.round(held_auc, 3).tolist()}", flush=True)
            _save(path, held=p["held"].astype(np.float32), idx=dev[te], loss=np.array(losses), seconds=sec)


def grid_table() -> tuple[list[dict], dict]:
    """Pooled out-of-fold development AUC of every (view, configuration, epochs), and the best per view."""
    b = load_bundle(signals=False)
    y_all = b.y
    rows, best = [], {}
    for view in VIEWS:
        for mode, lr in GRID:
            folder = WORK / "grid" / view / config_name(mode, lr)
            files = [folder / f"fold{k}.npz" for k in range(N_FOLDS)]
            if not all(f.exists() for f in files):
                continue
            z = [np.load(f) for f in files]
            idx = np.concatenate([q["idx"] for q in z])
            for ep in EPOCHS[mode]:
                s = np.concatenate([q["held"][ep - 1] for q in z])
                r = {"view": view, "mode": mode, "lr": lr, "epochs": ep, "dev_oof_auc": auc(s, y_all[idx]),
                     "fold_aucs": [auc(q["held"][ep - 1], y_all[q["idx"]]) for q in z],
                     "seconds": float(sum(q["seconds"] for q in z))}
                rows.append(r)
                # strictly better wins, so ties keep the earlier (frozen first, fewer epochs) configuration
                if view not in best or r["dev_oof_auc"] > best[view]["dev_oof_auc"]:
                    best[view] = r
    return rows, best


def select() -> None:
    rows, best = grid_table()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "selection.json").write_text(json.dumps({"best": best, "grid": rows}, indent=1))
    lines = ["| view | mode | lr | epochs | dev OOF AUC (pooled) | per-fold AUC | train s (5 folds, all epochs) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = " **(chosen)**" if best.get(r["view"]) is r else ""
        lines.append(f"| {r['view']} | {r['mode']} | {r['lr']:g} | {r['epochs']} | {r['dev_oof_auc']:.3f}{mark} | "
                     f"{', '.join(f'{a:.2f}' for a in r['fold_aucs'])} | {r['seconds']:.0f} |")
    (WORK / "grid.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def final(view: str) -> None:
    """The chosen configuration on each fold: held-out development predictions and test predictions."""
    import torch
    torch.backends.cudnn.benchmark = True
    sel = json.loads((WORK / "selection.json").read_text())["best"][view]
    mode, lr, n_ep = sel["mode"], float(sel["lr"]), int(sel["epochs"])
    b = load_bundle(signals=False)
    dev, test = b.dev, b.test
    fold, y = dev_folds(b), b.y[dev]
    imgs = load_images(b, dev, view, _device())
    imgs_test = load_images(b, test, view, _device())
    for k in range(N_FOLDS):
        path = WORK / "final" / view / f"fold{k}.npz"
        if path.exists():
            continue
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        t0 = time.time()
        print(f"final {view} {config_name(mode, lr)} {n_ep} epochs, fold {k}", flush=True)
        p, losses = train(view, mode, lr, n_ep, seed=k, imgs_tr=[x[tr] for x in imgs], y_tr=y[tr],
                          eval_sets={"held": [x[te] for x in imgs], "test": imgs_test}, eval_at={n_ep},
                          log=lambda s: print(s, flush=True))
        print(f"final {view} fold {k}: {time.time() - t0:.0f} s, held-out AUC {auc(p['held'][0], y[te]):.3f}",
              flush=True)  # development only; test predictions are saved, not looked at
        _save(path, held=p["held"][0].astype(np.float32), idx=dev[te], test=p["test"][0].astype(np.float32),
              test_idx=test, loss=np.array(losses), seconds=time.time() - t0)


def final_scores(b: Bundle, view: str) -> tuple[dict[str, float], dict[str, float]]:
    z = [np.load(WORK / "final" / view / f"fold{k}.npz") for k in range(N_FOLDS)]
    dev = {b.ids[i]: float(s) for q in z for i, s in zip(q["idx"], q["held"])}
    test_mean = np.mean([q["test"] for q in z], 0)
    test = {b.ids[i]: float(s) for i, s in zip(z[0]["test_idx"], test_mean)}
    assert len(dev) == len(b.dev) and len(test) == len(b.test)
    return dev, test


def finish_all() -> None:
    b = load_bundle(signals=False)
    sel = json.loads((WORK / "selection.json").read_text())
    common = ("Inception-V3 pretrained on ImageNet, transfer learning on the rendered ECG images (after Xiao et al., "
              "AMIA Jt Summits Transl Sci Proc 2018;2017:256-262); native-resolution letterbox, shifts <=16/8 px, "
              "AdamW batch 16, pos-weighted BCE; configuration chosen by record-grouped 5-fold CV AUC on development; "
              "development values are out-of-fold predictions of the final fold models; test score = mean of the "
              "5 fold models' probabilities")
    scores, results = {}, {}
    for view in VIEWS:
        if not (WORK / "final" / view / f"fold{N_FOLDS - 1}.npz").exists():
            print(f"{view}: no final models, skipped")
            continue
        best = sel["best"][view]
        dev, test = final_scores(b, view)
        scores[view] = (dev, test)
        cfg = {"view": view, "images": list(VIEWS[view]), "mode": best["mode"], "lr": best["lr"],
               "epochs": best["epochs"], "batch": BATCH, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
               "canvas": {k: CANVAS[k] for k in VIEWS[view]}, "max_shift_px": SHIFT,
               "grid_dev_oof_auc": best["dev_oof_auc"]}
        notes = (f"{common}. View {'+'.join(VIEWS[view])}: {best['mode']} "
                 f"{'backbone + linear head' if best['mode'] == 'frozen' else 'fine-tuning of all layers'}, "
                 f"lr {best['lr']:g}, {best['epochs']} epochs (grid CV AUC {best['dev_oof_auc']:.3f})")
        extra = {"config": cfg, "grid": [r for r in sel["grid"] if r["view"] == view]}
        results[view] = (notes, extra)
    if all(v in scores for v in FUSED):
        dev = {c: float(np.mean([scores[v][0][c] for v in FUSED])) for c in scores[FUSED[0]][0]}
        test = {c: float(np.mean([scores[v][1][c] for v in FUSED])) for c in scores[FUSED[0]][1]}
        scores["fusion"] = (dev, test)
        results["fusion"] = (f"{common}. Fusion fixed before training: unweighted mean of the {', '.join(FUSED)} "
                             f"models' probabilities (each with its own chosen configuration)",
                             {"config": {"fusion": "mean of probabilities", "views": list(FUSED),
                                         "per_view": {v: results[v][1]["config"] for v in FUSED}}})
    dev_auc = {v: auc(np.array([s[0][c] for c in s[0]]), np.array([b.y[b.ids.index(c)] for c in s[0]]))
               for v, s in scores.items()}
    main = max((v for v in MAIN_CANDIDATES if v in dev_auc), key=lambda v: dev_auc[v], default=None)
    for v, (notes, extra) in results.items():
        extra = extra | {"main_row": v == main,
                         "main_row_rule": f"best development CV AUC among {', '.join(MAIN_CANDIDATES)}"}
        if v == main:
            notes += "; main image-CNN row (best development CV AUC of the views and the fusion)"
        r = finish(f"imagecnn_{v}", b, scores[v][0], scores[v][1], notes, extra)
        print(f"imagecnn_{v}: dev AUC {r['splits']['dev']['auc']:.3f} test AUC {r['splits']['test']['auc']:.3f}"
              f"{'  (main row)' if v == main else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["grid", "select", "final", "finish"])
    ap.add_argument("--view", choices=list(VIEWS))
    ap.add_argument("--smoke", action="store_true", help="grid: one configuration, one fold, 2 epochs (timing)")
    a = ap.parse_args()
    if a.step in ("grid", "final") and not a.view:
        ap.error("--view is required")
    {"grid": lambda: grid(a.view, a.smoke), "select": select, "final": lambda: final(a.view),
     "finish": finish_all}[a.step]()

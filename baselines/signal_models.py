"""B2: a supervised 1-D CNN trained from scratch on the raw signal (`resnet1d`, ablation `resnet1d_noref`), plus the
GPU-run helpers shared with baselines/fm_baselines.py (fold results on disk, grid tables, selection).

Architecture: xresnet1d50, the 1-D XResNet of the PTB-XL benchmark (Strodthoff, Wagner, Schaeffter & Samek, IEEE JBHI
2021, "Deep learning for ECG analysis: benchmarks and insights from PTB-XL"), i.e. ResNet-50 with the "bag of tricks"
changes of He et al. (CVPR 2019, ResNet-D) in 1-D: a stem of three convolutions (32, 32, 64 channels, the first with
stride 2) and a max pool; bottleneck blocks [3, 4, 6, 3] with expansion 4 (inner widths 64, 128, 256, 512), the
stride in the middle convolution, average pooling before the 1x1 convolution of the shortcut and the last BatchNorm
of every block initialised to zero; kernel size 5 throughout (the benchmark's 1-D setting); the benchmark head:
concatenated average and max pooling, BN-dropout(0.25)-linear(128)-ReLU, BN-dropout(0.5)-linear(1). Written here
from those papers; no code of the PTB-XL benchmark repository (GPL-3.0) is used.

Input: one of the 8 current 10-s segments of a candidate stacked with the reference segment as extra channels
(6 channels: 3 current + 3 reference leads; a missing third lead stays zero); `resnet1d_noref`: the current segment
only (3 channels). Per-lead standardisation with the mean and SD of the real leads of the training folds (one pair
of constants per lead position, shared by current and reference, so that the two stay comparable). The candidate
score is the mean of the 8 per-segment probabilities. The raw bundle signal is used as is (notch-filtered, no
baseline or band-pass filtering), as the benchmark used raw PTB-XL; 100 Hz versions are resampled with a polyphase
filter (scipy.signal.resample_poly).

Training: binary cross-entropy without class weights (297 of 587 development candidates are ischemic), AdamW (weight
decay 0.01, betas 0.9/0.99) with the one-cycle schedule of fastai's fit_one_cycle (25% warm-up from lr/25, cosine
annealing to lr/25e5, momentum 0.95-0.85-0.95), batch 128 segment pairs (last partial batch dropped), bf16 autocast,
no augmentation, one seed per fold (the fold index). Pre-specified grid, chosen by the pooled out-of-fold development
AUC: sampling rate {100 Hz (benchmark default), 250 Hz} x peak learning rate {1e-3, 3e-3} x epochs {10, 25, 50}
(each epoch budget a separate one-cycle run); `resnet1d_noref` has its own selection on the same grid. The final
configuration is retrained on each of the 5 folds (same seeds); the development score is out-of-fold and the test
score the mean of the 5 fold models, computed once.

    python -m baselines.signal_models grid --variant ref --worker 0 --workers 2   # on the GPU (BL_RUNS=~/bl_runs)
    python -m baselines.signal_models final --variant ref                         # selects, retrains, scores test
    python -m baselines.signal_models table                                       # grid tables (development only)
    python -m baselines.signal_models finish                                      # locally: outputs/baselines/<name>/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baselines.common import OUT, auc
from baselines.learned_common import Bundle, dev_folds, finish, load_bundle

RUNS = Path(os.environ.get("BL_RUNS", str(OUT / "_signal_runs"))).expanduser()
K_SEG = 8
N_FOLDS = 5


# ----------------------------------------------------------------------------------------------- shared helpers

@dataclass
class Split:
    dev: np.ndarray    # bundle indices of the development candidates (b.dev)
    test: np.ndarray   # bundle indices of the test candidates
    fold: np.ndarray   # fold (0-4) of every development candidate, aligned with dev
    y: np.ndarray      # label of every candidate, float32

    def train_val(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.dev[self.fold != k], self.dev[self.fold == k]


def make_split(b: Bundle) -> Split:
    return Split(b.dev, b.test, dev_folds(b), b.y.astype(np.float32))


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_fold(path: Path, idx: np.ndarray, score: np.ndarray, epochs: list[int], test_idx: np.ndarray | None = None,
              test_score: np.ndarray | None = None, **info) -> None:
    """Out-of-fold scores of one fold model: score [len(epochs), len(idx)] (one row per evaluated epoch);
    test scores only for a final configuration. Written atomically, so a preempted job leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if test_idx is None else {"test_idx": test_idx, "test_score": test_score}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, idx=idx, score=np.atleast_2d(score), epochs=np.array(epochs), info=json.dumps(info), **extra)
    tmp.rename(path)


def grid_rows(root: Path, sp: Split) -> list[dict]:
    """Pooled out-of-fold development AUC of every (configuration, epoch) under root/<config>/fold<k>.npz."""
    pos = {int(i): j for j, i in enumerate(sp.dev)}
    yd = sp.y[sp.dev].astype(bool)
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        files = [d / f"fold{k}.npz" for k in range(N_FOLDS)]
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        if not all(f.exists() for f in files):
            rows.append({"config": d.name, **cfg, "complete": False, "folds_done": sum(f.exists() for f in files)})
            continue
        oof: dict[int, np.ndarray] = {}
        for f in files:
            z = np.load(f)
            for j, e in enumerate(z["epochs"]):
                oof.setdefault(int(e), np.full(len(sp.dev), np.nan))[[pos[int(i)] for i in z["idx"]]] = z["score"][j]
        for e, s in sorted(oof.items()):
            fold_auc = [auc(s[sp.fold == k], yd[sp.fold == k]) for k in range(N_FOLDS)]
            rows.append({"config": d.name, **cfg, "epochs": e, "complete": True, "auc": auc(s, yd),
                         "mean_fold_auc": float(np.mean(fold_auc)), "fold_auc": [round(a, 4) for a in fold_auc]})
    return rows


def best_row(rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("complete")]
    if not done or len(done) < len(rows):
        raise RuntimeError(f"grid incomplete: {[r['config'] for r in rows if not r.get('complete')]}")
    return max(done, key=lambda r: r["auc"])  # ties: the first row (configurations in name order)


def rows_markdown(rows: list[dict], keys: list[str]) -> str:
    head = "| " + " | ".join(keys + ["epochs", "dev OOF AUC", "mean fold AUC"]) + " |"
    lines = [head, "|" + "---|" * (len(keys) + 3)]
    for r in rows:
        if r.get("complete"):
            lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys)
                         + f" | {r['epochs']} | {r['auc']:.3f} | {r['mean_fold_auc']:.3f} |")
    return "\n".join(lines)


def final_scores(root: Path, sp: Split) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold development scores and the test score (mean of the fold models) of a final configuration."""
    pos = {int(i): j for j, i in enumerate(sp.dev)}
    dev = np.full(len(sp.dev), np.nan)
    tests = []
    for k in range(N_FOLDS):
        z = np.load(root / f"fold{k}.npz")
        assert list(z["test_idx"]) == list(sp.test)
        dev[[pos[int(i)] for i in z["idx"]]] = z["score"][-1]
        tests.append(z["test_score"])
    assert np.isfinite(dev).all()
    return dev, np.mean(tests, axis=0)


# ------------------------------------------------------------------------------------------------- xresnet1d50

def build_xresnet1d50(c_in: int, ks: int = 5, lin: int = 128, ps: float = 0.5):
    import torch
    from torch import nn

    def conv_bn(ni: int, nf: int, k: int, stride: int = 1, act: bool = True, zero_bn: bool = False) -> nn.Sequential:
        conv = nn.Conv1d(ni, nf, k, stride=stride, padding=k // 2, bias=False)
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        bn = nn.BatchNorm1d(nf)
        nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
        return nn.Sequential(conv, bn, nn.ReLU(inplace=True)) if act else nn.Sequential(conv, bn)

    class Bottleneck(nn.Module):
        def __init__(self, ni: int, nh: int, stride: int):
            super().__init__()
            nf = 4 * nh
            self.convs = nn.Sequential(conv_bn(ni, nh, 1), conv_bn(nh, nh, ks, stride),
                                       conv_bn(nh, nf, 1, act=False, zero_bn=True))
            idp: list[nn.Module] = []
            if stride != 1:
                idp.append(nn.AvgPool1d(stride, ceil_mode=True))
            if ni != nf:
                idp.append(conv_bn(ni, nf, 1, act=False))
            self.idpath = nn.Sequential(*idp)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.convs(x) + self.idpath(x))

    class ConcatPool(nn.Module):
        def forward(self, x):
            return torch.cat([x.mean(-1), x.amax(-1)], 1)

    layers, widths = (3, 4, 6, 3), (16, 64, 128, 256, 512)  # widths[0] * 4 = 64 = stem output
    blocks: list[nn.Module] = []
    for s, n in enumerate(layers):
        for j in range(n):
            ni = 4 * (widths[s] if j == 0 else widths[s + 1])
            blocks.append(Bottleneck(ni, widths[s + 1], stride=(1 if s == 0 else 2) if j == 0 else 1))
    nf = 4 * widths[-1]
    return nn.Sequential(
        conv_bn(c_in, 32, ks, 2), conv_bn(32, 32, ks), conv_bn(32, 64, ks), nn.MaxPool1d(ks, 2, ks // 2),
        *blocks,
        ConcatPool(), nn.BatchNorm1d(2 * nf), nn.Dropout(ps / 2), nn.Linear(2 * nf, lin), nn.ReLU(inplace=True),
        nn.BatchNorm1d(lin), nn.Dropout(ps), nn.Linear(lin, 1))


# ------------------------------------------------------------------------------------------------ B2 training

GRID = {"fs": (100, 250), "lr": (1e-3, 3e-3), "epochs": (10, 25, 50)}
BATCH, WD = 128, 1e-2
VARIANTS = {"ref": "resnet1d", "noref": "resnet1d_noref"}


def config_name(variant: str, fs: int, lr: float, epochs: int) -> str:
    return f"{variant}_fs{fs}_lr{lr:g}_ep{epochs}"


class Signals:
    """The bundle signals on the GPU at one sampling rate."""

    def __init__(self, b: Bundle, fs: int):
        import torch
        cur, ref = b.cur, b.ref
        if fs != 250:
            from scipy.signal import resample_poly
            g = math.gcd(fs, 250)
            cur = resample_poly(cur, fs // g, 250 // g, axis=-1).astype(np.float32)
            ref = resample_poly(ref, fs // g, 250 // g, axis=-1).astype(np.float32)
        self.np_cur, self.np_ref, self.np_mask = cur, ref, b.lead_mask
        dev = device()
        self.cur = torch.from_numpy(np.ascontiguousarray(cur)).to(dev)
        self.ref = torch.from_numpy(np.ascontiguousarray(ref)).to(dev)
        self.mask = torch.from_numpy(b.lead_mask.astype(np.float32)).to(dev)
        self.y = torch.from_numpy(b.y.astype(np.float32)).to(dev)

    def stats(self, idx: np.ndarray):
        """Per-lead mean and SD over the real leads of the current and reference segments of candidates idx."""
        import torch
        mu, sd = np.zeros(3, np.float32), np.ones(3, np.float32)
        for lead in range(3):
            ok = idx[self.np_mask[idx, lead]]
            if len(ok):
                v = np.concatenate([self.np_cur[ok, :, lead].ravel(), self.np_ref[ok, lead].ravel()])
                mu[lead], sd[lead] = v.mean(), v.std() + 1e-6
        dev = self.cur.device
        return torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)

    def inputs(self, i, k, use_ref: bool, mu, sd):
        """Standardised network input [B, 6 or 3, T] for candidates i (tensor) and segments k (tensor)."""
        import torch
        m = self.mask[i][:, :, None]
        x = (self.cur[i, k] - mu[None, :, None]) / sd[None, :, None] * m
        if use_ref:
            r = (self.ref[i] - mu[None, :, None]) / sd[None, :, None] * m
            x = torch.cat([x, r], 1)
        return x


def fit_resnet(sig: Signals, tr: np.ndarray, use_ref: bool, lr: float, epochs: int, seed: int):
    import torch
    import torch.nn.functional as F
    set_seed(seed)
    dev = device()
    model = build_xresnet1d50(6 if use_ref else 3).to(dev)
    mu, sd = sig.stats(tr)
    pairs = np.array([(i, k) for i in tr for k in range(K_SEG)])
    steps = len(pairs) // BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.25,
                                                div_factor=25.0, final_div_factor=1e5, base_momentum=0.85,
                                                max_momentum=0.95)
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        perm, tot = rng.permutation(len(pairs)), 0.0
        for s in range(steps):
            bt = torch.from_numpy(pairs[perm[s * BATCH:(s + 1) * BATCH]]).to(dev)
            x = sig.inputs(bt[:, 0], bt[:, 1], use_ref, mu, sd)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = model(x).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), sig.y[bt[:, 0]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == epochs:
            log(f"  epoch {ep + 1}/{epochs} train loss {tot / steps:.4f}")
    return model, (mu, sd)


def predict_resnet(model, sig: Signals, idx: np.ndarray, use_ref: bool, norm) -> np.ndarray:
    """Candidate score = mean over the 8 segments of the per-segment probability."""
    import torch
    model.eval()
    dev = device()
    out = []
    with torch.no_grad():
        for c in range(0, len(idx), 32):
            i = torch.from_numpy(np.repeat(idx[c:c + 32], K_SEG)).to(dev)
            k = torch.arange(K_SEG, device=dev).repeat(len(idx[c:c + 32]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                p = torch.sigmoid(model(sig.inputs(i, k, use_ref, *norm)).squeeze(1).float())
            out.append(p.view(-1, K_SEG).mean(1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def grid_jobs(variant: str) -> list[tuple[str, dict, int]]:
    jobs = []
    for fs in GRID["fs"]:
        for lr in GRID["lr"]:
            for ep in GRID["epochs"]:
                cfg = {"variant": variant, "fs": fs, "lr": lr, "epochs_budget": ep}
                jobs += [(config_name(variant, fs, lr, ep), cfg, k) for k in range(N_FOLDS)]
    return jobs


def run_grid(variants: list[str], worker: int, workers: int) -> None:
    b = load_bundle()
    sp = make_split(b)
    jobs = [j for v in variants for j in grid_jobs(v)]
    mine = [j for n, j in enumerate(jobs) if n % workers == worker]
    sigs: dict[int, Signals] = {}
    for name, cfg, k in mine:
        out = RUNS / "resnet1d" / "grid" / name
        if (out / f"fold{k}.npz").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg))
        sig = sigs.setdefault(cfg["fs"], Signals(b, cfg["fs"]))
        tr, va = sp.train_val(k)
        t0 = time.time()
        log(f"{name} fold {k}: train {len(tr)} candidates, held out {len(va)}")
        model, norm = fit_resnet(sig, tr, cfg["variant"] == "ref", cfg["lr"], cfg["epochs_budget"], seed=k)
        s = predict_resnet(model, sig, va, cfg["variant"] == "ref", norm)
        save_fold(out / f"fold{k}.npz", va, s, [cfg["epochs_budget"]], seconds=time.time() - t0)
        log(f"{name} fold {k}: held-out AUC {auc(s, sp.y[va].astype(bool)):.3f} ({time.time() - t0:.0f} s)")


def run_final(variant: str) -> None:
    """Select the configuration by the pooled out-of-fold development AUC, retrain it on each fold, score test."""
    b = load_bundle()
    sp = make_split(b)
    rows = [r for r in grid_rows(RUNS / "resnet1d" / "grid", sp) if r.get("variant") == variant]
    best = best_row(rows)
    out = RUNS / "resnet1d" / "final" / variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected.json").write_text(json.dumps({"selected": best, "grid": rows}, indent=1))
    log(f"{variant}: selected {best['config']} (dev OOF AUC {best['auc']:.3f})")
    sig = Signals(b, best["fs"])
    for k in range(N_FOLDS):
        if (out / f"fold{k}.npz").exists():
            continue
        tr, va = sp.train_val(k)
        t0 = time.time()
        model, norm = fit_resnet(sig, tr, variant == "ref", best["lr"], best["epochs_budget"], seed=k)
        s = predict_resnet(model, sig, va, variant == "ref", norm)
        st = predict_resnet(model, sig, sp.test, variant == "ref", norm)
        save_fold(out / f"fold{k}.npz", va, s, [best["epochs_budget"]], test_idx=sp.test, test_score=st,
                  seconds=time.time() - t0)
        log(f"final {variant} fold {k}: held-out AUC {auc(s, sp.y[va].astype(bool)):.3f}")


def table() -> None:
    sp = make_split(load_bundle(signals=False))
    rows = grid_rows(RUNS / "resnet1d" / "grid", sp)
    for v in VARIANTS:
        print(f"\n## {VARIANTS[v]}\n")
        print(rows_markdown([r for r in rows if r.get("variant") == v], ["fs", "lr"]))


def finish_all() -> None:
    """Write outputs/baselines/resnet1d{,_noref}/ from the final runs (development OOF, test = fold-model mean)."""
    b = load_bundle(signals=False)
    sp = make_split(b)
    for variant, name in VARIANTS.items():
        root = RUNS / "resnet1d" / "final" / variant
        if not (root / f"fold{N_FOLDS - 1}.npz").exists():
            print(f"{name}: final run missing")
            continue
        sel = json.loads((root / "selected.json").read_text())
        best = sel["selected"]
        dev, test = final_scores(root, sp)
        grid = [{k: r[k] for k in ("config", "fs", "lr", "epochs", "auc", "mean_fold_auc")} for r in sel["grid"]]
        inp = ("one current 10-s segment + the reference segment (6 channels)" if variant == "ref"
               else "one current 10-s segment only (3 channels)")
        notes = (f"xresnet1d50 (PTB-XL benchmark architecture, own implementation) trained from scratch on {inp}; "
                 f"per-lead standardisation (training-fold statistics); candidate score = mean of the 8 per-segment "
                 f"probabilities. Selected by pooled out-of-fold development AUC over fs {GRID['fs']} x lr "
                 f"{GRID['lr']} x epochs {GRID['epochs']}: {best['fs']} Hz, lr {best['lr']:g}, {best['epochs']} "
                 f"epochs (grid AUC {best['auc']:.3f}); AdamW one-cycle, wd {WD}, batch {BATCH}. The selected "
                 f"configuration was retrained on each fold (same seeds): development values are out-of-fold from "
                 f"these models, test = mean of the 5 fold models.")
        extra = {"config": {"architecture": "xresnet1d50", "input": inp, "fs": best["fs"], "lr": best["lr"],
                            "epochs": best["epochs"], "batch": BATCH, "weight_decay": WD,
                            "schedule": "one-cycle (pct_start 0.25, div 25, final div 1e5)", "seeds": "fold index"},
                 "grid": grid}
        r = finish(name, b, {b.ids[i]: float(s) for i, s in zip(sp.dev, dev)},
                   {b.ids[i]: float(s) for i, s in zip(sp.test, test)}, notes, extra=extra)
        print(name, {s: round(d["auc"], 3) for s, d in r["splits"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["grid", "final", "table", "finish"])
    ap.add_argument("--variant", choices=list(VARIANTS), action="append")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "grid":
        run_grid(a.variant or list(VARIANTS), a.worker, a.workers)
    elif a.cmd == "final":
        for v in a.variant or list(VARIANTS):
            run_final(v)
    elif a.cmd == "table":
        table()
    else:
        finish_all()

"""B4: ECG foundation models, as frozen feature extractors (linear probe) and fully fine-tuned, on the candidates and
protocol of baselines/learned_common.py.

Models: the authors' code at pinned commits (cloned on the pod by sky/bl_signal_setup.sh; imported, not copied) and
their released weights.
- ECG-JEPA (Kim, arXiv 2410.08559; github.com/sehunfromdaegu/ECG_JEPA @ d937ad2, MIT): a ViT encoder (768-d, 12
  layers, 16 heads) over 50-sample patches of 8 leads (I, II, V1-V6) at 250 Hz, 10 s, mV, no normalisation; cross-
  pattern attention; representation = mean of the output tokens (models.load_encoder, representation()). Two
  checkpoints: random masking (random_epoch100.pth) and multi-block masking (multiblock_epoch100.pth).
- ECGFounder (Li et al., NEJM AI 2025; github.com/PKUDigitalHealth/ECGFounder @ 04edac7, MIT): Net1D, a RegNet-style
  1-D CNN, 500 Hz, 10 s; a 12-lead model (I, II, III, aVR, aVL, aVF, V1-V6) and a single-lead model (lead I);
  representation = time average of the last stage (1024-d, the input of the classifier `dense`).

Preprocessing, as each repository expects:
- ECG-JEPA: the bundle signal (250 Hz, mV, notch only) as is for the probe (downstream_tasks/linear_eval.py feeds raw
  waves); for fine-tuning also the repository's fine-tuning filters, 0.67 Hz high-pass then 40 Hz low-pass, 5th-order
  Butterworth, zero phase (downstream_tasks/finetuning.py, augmentation.SOSFilter); its RandAugment is not used.
- ECGFounder: its README requires dataset.py's preprocessing with the released weights: linear interpolation to
  500 Hz (dataset.resample_unequal); util.filter_bandpass: 50 Hz notch (Q 30) and 0.67-40 Hz 4th-order Butterworth
  band-pass, both filtfilt, minus a 0.4-s median-filter baseline; a z-score over the whole segment (all real leads
  together; for the single-lead model the lead alone); missing leads zero after the z-score. The paper describes
  another pipeline (0.5 Hz high-pass, 2nd-order 50 Hz low-pass, 50/60 Hz notch, per-segment z-score); its probe is
  a development-only sensitivity analysis (embedding `founder_paper_*`), never scored on test.

Lead mapping, two pre-specified options chosen by development CV (both reported):
  (a) name mapping into the model's slots: ML2 -> II, MLIII -> III, MV2 -> V2; I, II, III, aVF, V1-V6 by name. Leads
      without a slot (E-S, A-S, A-I; MLIII and aVF for ECG-JEPA) take, in record order, the first free slot of the
      fixed order ECG-JEPA: II, I, V1, V2, V3, V4, V5, V6; ECGFounder: II, I, III, aVF, aVL, aVR, V1, ..., V6.
      ECGFounder: unused slots are zero (12-lead model). ECG-JEPA: only the tokens of the occupied slots are fed,
      with those slots' positional embeddings and attention mask: the repository's reduced-lead mechanism
      (MaskTransformer.representation with `leads`, the paper's single/dual-lead evaluation) instead of zero slots.
  (b) per lead: every real lead alone (ECGFounder single-lead model; ECG-JEPA with the lead in the II slot); segment
      embedding = mean over the leads.

Linear probe: frozen encoder (fp32); e_cur = mean over the 8 current segments, e_ref from the reference segment;
features [e_cur, e_ref, e_cur - e_ref] standardised on the training folds; L2 logistic regression (scikit-learn,
lbfgs) with C in {1e-4, 3e-4, ..., 3, 10}. Mapping (and the ECG-JEPA checkpoint) and C are chosen by the pooled
out-of-fold development AUC; test = mean of the 5 fold models. `ecgjepa_random_probe`: the same pipeline on the
ECG-JEPA architecture with the repository's random initialisation (seed 0); mapping and C by its own CV.

Fine-tuning: encoder + linear head on [h_cur, h_ref, h_cur - h_ref] for one (current segment, reference segment)
pair, the encoder shared by both; binary cross-entropy; all parameters at one learning rate (as in both
repositories); batch 32 pairs; linear warm-up over 50 steps, then constant; ECG-JEPA: AdamW, betas (0.9, 0.95),
weight decay 0.05 (its fine-tuning recipe); ECGFounder: Adam, weight decay 1e-5 (its fine-tuning notebook); bf16
autocast; candidate score = mean of the 8 per-segment probabilities. Grid: checkpoint {multi-block, random}
(ECG-JEPA) x mapping {a, b} x learning rate {1e-5, 3e-5, 1e-4} x epochs {1, 2, 4, 8}. With a constant learning rate
the model after e epochs of a run is the model of an e-epoch run, so the held-out fold is scored after epochs 1, 2,
4 and 8 of one run; the epoch count is chosen with the other hyperparameters by the pooled out-of-fold AUC (one value
for all folds, never per fold). The chosen configuration is retrained on each fold (same seeds) for that many epochs
and then scores its held-out fold and, once, the test set.

    python -m baselines.fm_baselines check                       # GPU: code checks, parameter counts, timing
    python -m baselines.fm_baselines embed                       # GPU: frozen embeddings of every candidate
    python -m baselines.fm_baselines probe                       # CPU: probes (development CV, selection, test)
    python -m baselines.fm_baselines ft-grid --worker 0 --workers 2
    python -m baselines.fm_baselines ft-final --model jepa       # selects, retrains, scores test
    python -m baselines.fm_baselines table                       # development grids
    python -m baselines.fm_baselines finish                      # locally: outputs/baselines/<name>/
    python -m baselines.fm_baselines finish-paperprep            # locally: extra row ecgfounder_probe_paperprep
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from baselines.common import auc
from baselines.learned_common import Bundle, finish, load_bundle
from baselines.signal_models import (K_SEG, N_FOLDS, RUNS, Split, best_row, device, final_scores, grid_rows, log,
                                     make_split, rows_markdown, save_fold, set_seed)

TP = Path(os.environ.get("THIRD_PARTY", "~/third_party")).expanduser()
WEIGHTS = Path(os.environ.get("FM_WEIGHTS", "~/fm_weights")).expanduser()
JEPA_CKPT = {"multiblock": "multiblock_epoch100.pth", "random": "random_epoch100.pth"}
FOUNDER_CKPT = {"a": "12_lead_ECGFounder.pth", "b": "1_lead_ECGFounder.pth"}
FM = RUNS / "fm"

JEPA_SLOTS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
FOUNDER_SLOTS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ALIAS = {"ML2": "II", "MLII": "II", "MLIII": "III", "MV2": "V2"}
FREE_ORDER = {"jepa": ("II", "I", "V1", "V2", "V3", "V4", "V5", "V6"),
              "founder": ("II", "I", "III", "aVF", "aVL", "aVR", "V1", "V2", "V3", "V4", "V5", "V6")}

C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0)
FT_LR, FT_EPOCHS, FT_BATCH, WARMUP = (1e-5, 3e-5, 1e-4), (1, 2, 4, 8), 32, 50
EMBEDDINGS = [  # (name, family, checkpoint, mapping, preprocessing)
    ("jepa_multiblock_a", "jepa", "multiblock", "a", "raw"), ("jepa_multiblock_b", "jepa", "multiblock", "b", "raw"),
    ("jepa_random_a", "jepa", "random", "a", "raw"), ("jepa_random_b", "jepa", "random", "b", "raw"),
    ("jepa_init_a", "jepa", "init", "a", "raw"), ("jepa_init_b", "jepa", "init", "b", "raw"),
    ("founder_a", "founder", "12lead", "a", "repo"), ("founder_b", "founder", "1lead", "b", "repo"),
    ("founder_paper_a", "founder", "12lead", "a", "paper"), ("founder_paper_b", "founder", "1lead", "b", "paper")]
PROBES = {"ecgjepa_probe": ["jepa_multiblock_a", "jepa_multiblock_b", "jepa_random_a", "jepa_random_b"],
          "ecgjepa_random_probe": ["jepa_init_a", "jepa_init_b"],
          "ecgfounder_probe": ["founder_a", "founder_b"],
          "founder_paper_sensitivity": ["founder_paper_a", "founder_paper_b"]}  # development only
FT_NAMES = {"jepa": "ecgjepa_ft", "founder": "ecgfounder_ft"}


# ------------------------------------------------------------------------------------------------ lead mapping

def slot_map(leads: list[str], family: str) -> tuple[int, ...]:
    """Model slot of every real lead (option a)."""
    slots = JEPA_SLOTS if family == "jepa" else FOUNDER_SLOTS
    out: list[int | None] = [None] * len(leads)
    used: set[int] = set()
    for j, name in enumerate(leads):
        s = ALIAS.get(name, name)
        if s in slots and slots.index(s) not in used:
            out[j] = slots.index(s)
            used.add(out[j])
    for j in range(len(leads)):
        if out[j] is None:
            out[j] = next(slots.index(f) for f in FREE_ORDER[family] if slots.index(f) not in used)
            used.add(out[j])
    return tuple(out)


def lead_slots(b: Bundle, family: str) -> np.ndarray:
    """[N, 3] slot of each lead position of every candidate (-1 for a missing lead)."""
    out = np.full((len(b.ids), 3), -1, int)
    for i, cid in enumerate(b.ids):
        n = int(b.meta[cid]["n_leads"])
        out[i, :n] = slot_map(b.meta[cid]["leads"][:n], family)
    return out


def mapping_table(b: Bundle) -> dict[str, dict[str, str]]:
    tab = {}
    for cid in b.ids:
        leads = tuple(b.meta[cid]["leads"][:int(b.meta[cid]["n_leads"])])
        tab[" ".join(leads)] = {f: " ".join(f"{l}->{(JEPA_SLOTS if f == 'jepa' else FOUNDER_SLOTS)[s]}"
                                            for l, s in zip(leads, slot_map(list(leads), f)))
                                for f in ("jepa", "founder")}
    return tab


# -------------------------------------------------------------------------------------------- preprocessing

def jepa_ft_filter(x: np.ndarray) -> np.ndarray:
    """ECG-JEPA fine-tuning filters (finetuning.py): 0.67 Hz high-pass, then 40 Hz low-pass, order 5, zero phase."""
    from scipy.signal import butter, sosfiltfilt
    hp = butter(5, 0.67, btype="highpass", fs=250, output="sos")
    lp = butter(5, 40, btype="lowpass", fs=250, output="sos")
    return sosfiltfilt(lp, sosfiltfilt(hp, x.astype(np.float64), axis=-1), axis=-1).astype(np.float32)


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """scipy.signal.medfilt(row, k) of every row (zero padding at the ends), on the GPU if there is one."""
    import torch
    import torch.nn.functional as F
    dev, out = device(), np.empty_like(x)
    if dev.type == "cpu":
        from scipy.ndimage import median_filter as ndmedian
        return ndmedian(x, size=(1, k), mode="constant", cval=0.0)
    for c in range(0, len(x), 256):
        t = F.pad(torch.from_numpy(np.ascontiguousarray(x[c:c + 256])).to(dev), (k // 2, k // 2))
        out[c:c + 256] = t.unfold(-1, k, 1).median(-1).values.cpu().numpy()
    return out


def founder_prep(x: np.ndarray, kind: str) -> np.ndarray:
    """[..., 2500] at 250 Hz -> [..., 5000] at 500 Hz, filtered as ECGFounder expects (before the z-score)."""
    from scipy.interpolate import interp1d
    from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    up = interp1d(np.linspace(0, 10, 2500), flat, axis=-1)(np.linspace(0, 10, 5000))  # resample_unequal
    if kind == "repo":  # util.filter_bandpass(signal, 500)
        bn, an = iirnotch(50, 30, 500)
        up = filtfilt(bn, an, up, axis=-1)
        bb, ab = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=500)
        up = filtfilt(bb, ab, up, axis=-1)
        up = up - median_filter(up, int(0.4 * 500) + 1)
    elif kind == "paper":  # paper text: 0.5 Hz high-pass, 2nd-order 50 Hz low-pass, 50 Hz notch (orders as stated or 2)
        up = sosfiltfilt(butter(2, 0.5, btype="highpass", fs=500, output="sos"), up, axis=-1)
        up = sosfiltfilt(butter(2, 50, btype="lowpass", fs=500, output="sos"), up, axis=-1)
        bn, an = iirnotch(50, 30, 500)
        up = filtfilt(bn, an, up, axis=-1)
    else:
        raise ValueError(kind)
    return up.reshape(*x.shape[:-1], 5000).astype(np.float32)


def signals(b: Bundle, family: str, prep: str) -> tuple[np.ndarray, np.ndarray]:
    """(cur [N, 8, 3, T], ref [N, 3, T]) preprocessed for a model, cached under FM/prep."""
    if family == "jepa" and prep == "raw":
        return b.cur, b.ref
    path = FM / "prep" / f"{family}_{prep}.npy"
    if not path.exists():
        both = np.concatenate([b.cur, b.ref[:, None]], 1)  # [N, 9, 3, 2500]
        t0 = time.time()
        out = jepa_ft_filter(both) if family == "jepa" else founder_prep(both, prep)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".tmp.npy"), out)
        path.with_suffix(".tmp.npy").rename(path)
        log(f"preprocessed {family}/{prep} in {time.time() - t0:.0f} s")
    out = np.load(path)
    return out[:, :K_SEG], out[:, K_SEG]


# ------------------------------------------------------------------------------------------------- encoders

def load_pth(path: Path) -> dict:
    """torch.load(weights_only=True). The ECGFounder checkpoints also hold a numpy float64 (val_auroc), so the
    numpy scalar reconstructor and dtypes are allowed as well; nothing else is unpickled."""
    import numpy._core.multiarray as ma
    import torch
    safe = [(ma.scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


def jepa_encoder(ckpt: str):
    """ECG-JEPA encoder as models.load_encoder builds it; ckpt 'init' = the repository's random initialisation."""
    import torch
    if str(TP / "ECG_JEPA") not in sys.path:
        sys.path.insert(0, str(TP / "ECG_JEPA"))
    from ecg_jepa import ecg_jepa
    torch.manual_seed(0)
    enc = ecg_jepa(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=16, predictor_embed_dim=384,
                   predictor_depth=6, predictor_num_heads=12, c=8, pos_type="sincos", mask_scale=(0, 0),
                   leads=list(range(8))).encoder
    if ckpt != "init":
        enc.load_state_dict(load_pth(WEIGHTS / JEPA_CKPT[ckpt])["encoder"])
    return enc


def founder_net(mapping: str):
    """ECGFounder Net1D as finetune_model.ft_*_ECGFounder builds it, classifier replaced by the identity."""
    from torch import nn
    spec = importlib.util.spec_from_file_location("ecgfounder_net1d", TP / "ECGFounder" / "net1d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.Net1D(in_channels=12 if mapping == "a" else 1, base_filters=64, ratio=1,
                    filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                    kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False, use_do=False,
                    n_classes=150)
    sd = load_pth(WEIGHTS / FOUNDER_CKPT[mapping])["state_dict"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    res = net.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        log(f"ECGFounder {mapping}: missing {res.missing_keys[:5]} unexpected {res.unexpected_keys[:5]}")
    net.dense = nn.Identity()
    return net


_MASKS: dict = {}


def jepa_forward(enc, x, slots: tuple[int, ...]):
    """ECG-JEPA representation of x [B, L, 2500] whose leads sit in `slots`: MaskTransformer.representation with
    leads=slots (positional embeddings and cross-pattern attention mask restricted to those leads), batched."""
    import torch
    bsz, n_lead, _ = x.shape
    dim = enc.embed_dim
    key = (slots, str(x.device))
    if key not in _MASKS:
        s = torch.tensor(slots, device=x.device)
        full = enc._cross_attention_mask().to(x.device)
        _MASKS[key] = (s, full.view(8, 50, 8, 50)[s][:, :, s].reshape(n_lead * 50, n_lead * 50))
    s, mask = _MASKS[key]
    pos = enc.pos_embed.view(8, 50, dim)[s].reshape(n_lead * 50, dim)
    t = enc.W_P(x.reshape(bsz, n_lead * 50, 50))
    t = enc.norm(enc.encoder_blocks(t, pos, mask))
    return t.mean(1)


def make_encoder(family: str, ckpt: str, mapping: str, b: Bundle):
    import torch
    from torch import nn

    class SegmentEncoder(nn.Module):
        """Embedding [B, D] of segments x [B, 3, T] (bundle lead order) of candidates `cand` [B]."""

        def __init__(self):
            super().__init__()
            self.family, self.mapping = family, mapping
            self.net = jepa_encoder(ckpt) if family == "jepa" else founder_net(mapping)
            self.dim = 768 if family == "jepa" else 1024
            slots = lead_slots(b, family)
            self.register_buffer("mask", torch.from_numpy(b.lead_mask.copy()), persistent=False)
            self.register_buffer("slots", torch.from_numpy(np.maximum(slots, 0)), persistent=False)
            keys = [tuple(int(s) for s in row if s >= 0) for row in slots]
            self.key_list = sorted(set(keys))
            self.register_buffer("key", torch.tensor([self.key_list.index(k) for k in keys]), persistent=False)

        def per_lead(self, x, cand, fn):
            rows, leads = self.mask[cand].nonzero(as_tuple=True)
            e = fn(x[rows, leads]).float()
            out = torch.zeros(len(cand), e.shape[1], device=x.device).index_add(0, rows, e)
            return out / self.mask[cand].sum(1, keepdim=True).float()

        def forward(self, x, cand):
            if self.family == "jepa" and self.mapping == "a":
                out = torch.zeros(len(cand), self.dim, device=x.device)
                keys = self.key[cand]
                for kk in keys.unique().tolist():
                    sel = (keys == kk).nonzero().squeeze(1)
                    slots = self.key_list[kk]
                    out[sel] = jepa_forward(self.net, x[sel, :len(slots)], slots).float()
                return out
            if self.family == "jepa":
                return self.per_lead(x, cand, lambda v: jepa_forward(self.net, v[:, None], (1,)))
            if self.mapping == "b":
                return self.per_lead(x, cand, lambda v: self.net(
                    ((v - v.mean(1, keepdim=True)) / (v.std(1, unbiased=False, keepdim=True) + 1e-8))[:, None]))
            m = self.mask[cand].float()[:, :, None]
            n = m.sum((1, 2)) * x.shape[-1]
            mu = (x * m).sum((1, 2)) / n
            sd = ((((x - mu[:, None, None]) * m) ** 2).sum((1, 2)) / n).sqrt()
            xn = (x - mu[:, None, None]) / (sd[:, None, None] + 1e-8) * m
            x12 = x.new_zeros(len(cand), 12, x.shape[-1])
            rows, leads = self.mask[cand].nonzero(as_tuple=True)
            x12[rows, self.slots[cand[rows], leads]] = xn[rows, leads]
            return self.net(x12).float()

    return SegmentEncoder()


def make_pair_model(enc):
    import torch
    from torch import nn

    class PairModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.head = nn.Linear(3 * enc.dim, 1)

        def logits(self, hc, hr):
            return self.head(torch.cat([hc, hr, hc - hr], 1)).squeeze(1)

        def forward(self, xc, xr, cand):
            h = self.enc(torch.cat([xc, xr]), torch.cat([cand, cand]))
            hc, hr = h.chunk(2)
            return self.logits(hc, hr)

    return PairModel()


# ----------------------------------------------------------------------------------------------- embeddings

def extract(enc, cur, ref, chunk: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Frozen embeddings: cur [N, 8, D], ref [N, D] of every candidate (fp32)."""
    import torch
    dev = device()
    enc.eval()
    n = cur.shape[0]
    ec, er = np.zeros((n, K_SEG, enc.dim), np.float32), np.zeros((n, enc.dim), np.float32)
    with torch.no_grad():
        for c in range(0, n, chunk):
            idx = torch.arange(c, min(n, c + chunk), device=dev)
            xc = torch.from_numpy(cur[c:c + chunk]).to(dev).reshape(-1, 3, cur.shape[-1])
            ec[c:c + chunk] = enc(xc, idx.repeat_interleave(K_SEG)).view(len(idx), K_SEG, -1).cpu().numpy()
            er[c:c + chunk] = enc(torch.from_numpy(ref[c:c + chunk]).to(dev), idx).cpu().numpy()
    return ec, er


def run_embed(names: list[str] | None = None) -> None:
    b = load_bundle()
    for name, family, ckpt, mapping, prep in EMBEDDINGS:
        out = FM / "emb" / f"{name}.npz"
        if (names and name not in names) or out.exists():
            continue
        t0 = time.time()
        cur, ref = signals(b, family, prep)
        enc = make_encoder(family, ckpt, mapping, b).to(device())
        ec, er = extract(enc, cur, ref)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".tmp.npz"), cur=ec, ref=er, ids=np.array(b.ids))
        out.with_suffix(".tmp.npz").rename(out)
        log(f"embeddings {name}: {ec.shape} in {time.time() - t0:.0f} s; finite {np.isfinite(ec).all()}")
        del enc


# -------------------------------------------------------------------------------------------------- probes

def probe_features(z, idx: np.ndarray) -> np.ndarray:
    ec, er = z["cur"][idx].mean(1), z["ref"][idx]
    return np.hstack([ec, er, ec - er])


def probe_model(c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=20000))


def run_probe() -> None:
    b = load_bundle(signals=False)
    sp = make_split(b)
    yd = sp.y[sp.dev].astype(bool)
    for name, embs in PROBES.items():
        rows, best = [], None
        for emb in embs:
            z = dict(np.load(FM / "emb" / f"{emb}.npz"))
            assert list(z["ids"]) == b.ids
            X = probe_features(z, sp.dev)
            for c in C_GRID:
                oof = np.zeros(len(sp.dev))
                for k in range(N_FOLDS):
                    tr, va = sp.fold != k, sp.fold == k
                    oof[va] = probe_model(c).fit(X[tr], yd[tr]).predict_proba(X[va])[:, 1]
                fold_auc = [auc(oof[sp.fold == k], yd[sp.fold == k]) for k in range(N_FOLDS)]
                r = {"embedding": emb, "C": c, "auc": auc(oof, yd), "mean_fold_auc": float(np.mean(fold_auc))}
                rows.append(r)
                if best is None or r["auc"] > best[0]["auc"]:
                    best = (r, oof)
            log(f"{name} {emb}: best C AUC {max(r['auc'] for r in rows if r['embedding'] == emb):.3f}")
        r, oof = best
        res = {"selected": r, "grid": rows, "dev": oof.tolist()}
        if name != "founder_paper_sensitivity":  # sensitivity analysis: development only
            z = dict(np.load(FM / "emb" / f"{r['embedding']}.npz"))
            X, Xt = probe_features(z, sp.dev), probe_features(z, sp.test)
            res["test"] = np.mean([probe_model(r["C"]).fit(X[sp.fold != k], yd[sp.fold != k]).predict_proba(Xt)[:, 1]
                                   for k in range(N_FOLDS)], axis=0).tolist()
        (FM / "probe").mkdir(parents=True, exist_ok=True)
        (FM / "probe" / f"{name}.json").write_text(json.dumps(res, indent=1))
        log(f"{name}: selected {r['embedding']} C={r['C']:g} dev OOF AUC {r['auc']:.3f}")


# ------------------------------------------------------------------------------------------------ fine-tuning

def ft_configs(model: str) -> list[dict]:
    if model == "jepa":
        return [{"family": "jepa", "ckpt": c, "mapping": m, "lr": lr}
                for c in ("multiblock", "random") for m in "ab" for lr in FT_LR]
    return [{"family": "founder", "ckpt": "12lead/1lead", "mapping": m, "lr": lr} for m in "ab" for lr in FT_LR]


def ft_name(cfg: dict) -> str:
    ck = cfg["ckpt"] if cfg["family"] == "jepa" else ("12lead" if cfg["mapping"] == "a" else "1lead")
    return f"{cfg['family']}_{ck}_{cfg['mapping']}_lr{cfg['lr']:g}"


class FTData:
    """Preprocessed fine-tuning inputs of every candidate on the GPU."""

    def __init__(self, b: Bundle, family: str):
        import torch
        cur, ref = signals(b, family, "ft" if family == "jepa" else "repo")
        dev = device()
        self.cur, self.ref = torch.from_numpy(cur).to(dev), torch.from_numpy(ref).to(dev)
        self.y = torch.from_numpy(b.y.astype(np.float32)).to(dev)


def predict_ft(model, data: FTData, idx: np.ndarray, chunk: int = 32) -> np.ndarray:
    import torch
    dev = device()
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for c in range(0, len(idx), chunk):
            i = torch.from_numpy(idx[c:c + chunk]).to(dev)
            hr = model.enc(data.ref[i], i)
            hc = model.enc(data.cur[i].reshape(-1, *data.cur.shape[2:]), i.repeat_interleave(K_SEG))
            p = torch.sigmoid(model.logits(hc, hr.repeat_interleave(K_SEG, 0)).float())
            out.append(p.view(-1, K_SEG).mean(1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def fit_ft(cfg: dict, b: Bundle, data: FTData, tr: np.ndarray, va: np.ndarray, epochs: tuple[int, ...], seed: int,
           test: np.ndarray | None = None, max_steps: int | None = None) -> dict[int, tuple]:
    """Fine-tune on candidates tr; held-out (and test) scores after each epoch in `epochs`."""
    import torch
    import torch.nn.functional as F
    set_seed(seed)
    dev = device()
    model = make_pair_model(make_encoder(cfg["family"], cfg["ckpt"], cfg["mapping"], b)).to(dev)
    if cfg["family"] == "jepa":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.05)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    pairs = np.array([(i, k) for i in tr for k in range(K_SEG)])
    steps = len(pairs) // FT_BATCH
    rng = np.random.default_rng(seed)
    model.train()
    out, n_step = {}, 0
    for ep in range(1, max(epochs) + 1):
        perm, tot, t0 = rng.permutation(len(pairs)), 0.0, time.time()
        for s in range(steps):
            bt = torch.from_numpy(pairs[perm[s * FT_BATCH:(s + 1) * FT_BATCH]]).to(dev)
            i, k = bt[:, 0], bt[:, 1]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = model(data.cur[i, k], data.ref[i], i)
            loss = F.binary_cross_entropy_with_logits(logit.float(), data.y[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            n_step += 1
            if max_steps and n_step >= max_steps:
                return {"seconds_per_step": (time.time() - t0) / n_step}
        log(f"  epoch {ep}: train loss {tot / steps:.4f} ({time.time() - t0:.0f} s)")
        if ep in epochs:
            out[ep] = (predict_ft(model, data, va), None if test is None else predict_ft(model, data, test))
    return out


def run_ft_grid(worker: int, workers: int, models: list[str]) -> None:
    b = load_bundle()
    sp = make_split(b)
    jobs = [(cfg, k) for m in models for cfg in ft_configs(m) for k in range(N_FOLDS)]
    data: dict[str, FTData] = {}
    for n, (cfg, k) in enumerate(jobs):
        out = FM / "ft" / "grid" / ft_name(cfg)
        if n % workers != worker or (out / f"fold{k}.npz").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg))
        tr, va = sp.train_val(k)
        t0 = time.time()
        log(f"{ft_name(cfg)} fold {k}: train {len(tr)}, held out {len(va)}")
        d = data.setdefault(cfg["family"], FTData(b, cfg["family"]))
        res = fit_ft(cfg, b, d, tr, va, FT_EPOCHS, seed=k)
        save_fold(out / f"fold{k}.npz", va, np.stack([res[e][0] for e in FT_EPOCHS]), list(FT_EPOCHS),
                  seconds=time.time() - t0)
        yv = sp.y[va].astype(bool)
        log(f"{ft_name(cfg)} fold {k}: held-out AUC by epoch "
            f"{ {e: round(auc(res[e][0], yv), 3) for e in FT_EPOCHS} } ({time.time() - t0:.0f} s)")


def run_ft_final(model: str) -> None:
    b = load_bundle()
    sp = make_split(b)
    names = {ft_name(c) for c in ft_configs(model)}
    rows = [r for r in grid_rows(FM / "ft" / "grid", sp) if r["config"] in names]
    if len({r["config"] for r in rows}) < len(names):
        raise RuntimeError(f"grid incomplete for {model}")
    best = best_row(rows)
    out = FM / "ft" / "final" / model
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected.json").write_text(json.dumps({"selected": best, "grid": rows}, indent=1))
    log(f"{model}: selected {best['config']} epochs {best['epochs']} (dev OOF AUC {best['auc']:.3f})")
    data = FTData(b, model)
    cfg = {k: best[k] for k in ("family", "ckpt", "mapping", "lr")}
    for k in range(N_FOLDS):
        if (out / f"fold{k}.npz").exists():
            continue
        tr, va = sp.train_val(k)
        t0 = time.time()
        res = fit_ft(cfg, b, data, tr, va, (best["epochs"],), seed=k, test=sp.test)
        s, st = res[best["epochs"]]
        save_fold(out / f"fold{k}.npz", va, s, [best["epochs"]], test_idx=sp.test, test_score=st,
                  seconds=time.time() - t0)
        log(f"final {model} fold {k}: held-out AUC {auc(s, sp.y[va].astype(bool)):.3f}")


# ------------------------------------------------------------------------------------------------- checks

def run_check() -> None:
    """Code checks on the GPU: the batched restricted ECG-JEPA forward equals the repository's representation();
    GPU median filter equals scipy.signal.medfilt; checkpoints load; parameter counts; lead mapping; step timing."""
    import torch
    from scipy.signal import medfilt
    b = load_bundle()
    dev = device()
    print(json.dumps(mapping_table(b), indent=1))
    enc = jepa_encoder("multiblock").to(dev).eval()
    x = torch.from_numpy(b.cur[:4, 0]).to(dev)  # real signals, [4, 3, 2500]
    with torch.no_grad():
        for slots in [(1,), (1, 5), (0, 1, 2, 3, 4, 5, 6, 7)]:
            xs = x[:, :len(slots)] if len(slots) <= 3 else torch.from_numpy(
                np.random.default_rng(0).normal(0, 0.3, (4, 8, 2500)).astype(np.float32)).to(dev)
            enc.leads = list(slots)
            ref = enc.representation(xs)
            mine = jepa_forward(enc, xs, slots)
            log(f"ECG-JEPA restricted forward vs representation(), leads {slots}: max abs diff "
                f"{(ref - mine).abs().max().item():.2e} (scale {ref.abs().mean().item():.2e})")
        enc.leads = list(range(8))
    log(f"ECG-JEPA encoder parameters: {sum(p.numel() for p in enc.parameters()) / 1e6:.1f} M")
    for m in "ab":
        net = founder_net(m)
        log(f"ECGFounder {FOUNDER_CKPT[m]}: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters")
    rows = np.random.default_rng(1).normal(0, 1, (3, 5000))
    diff = np.abs(median_filter(rows, 201) - np.stack([medfilt(r, 201) for r in rows])).max()
    log(f"median filter ({dev.type}) vs scipy.signal.medfilt: max abs diff {diff:.2e}")
    sp = make_split(b)
    tr, va = sp.train_val(0)
    quick = dev.type == "cpu"  # CPU: a smoke test of the code paths only (3 steps, a small resnet epoch)
    for model in ("jepa", "founder"):
        data = FTData(b, model)
        for m in "ab":
            cfg = {"family": model, "ckpt": "multiblock" if model == "jepa" else "-", "mapping": m, "lr": 3e-5}
            r = fit_ft(cfg, b, data, tr, va, (1,), seed=0, max_steps=3 if quick else 20)
            steps = len(tr) * K_SEG // FT_BATCH
            log(f"fine-tuning {model}/{m}: {r['seconds_per_step']:.3f} s/step, ~{r['seconds_per_step'] * steps:.0f} s "
                f"per epoch (fold 0)")
        torch.cuda.empty_cache()
    from baselines.signal_models import Signals, fit_resnet, predict_resnet
    for fs in (100, 250):
        t0 = time.time()
        sig = Signals(b, fs)
        sub = tr[::12] if quick else tr
        model, norm = fit_resnet(sig, sub, True, 1e-3, 1, seed=0)
        s = predict_resnet(model, sig, va[:20], True, norm)
        log(f"xresnet1d50 {fs} Hz: one epoch on {len(sub)} candidates {time.time() - t0:.1f} s; "
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters; scores finite {np.isfinite(s).all()}")
    if quick:
        r = fit_ft({"family": "jepa", "ckpt": "init", "mapping": "a", "lr": 3e-5}, b, FTData(b, "jepa"), tr[::12],
                   va[:10], (1,), seed=0, test=va[10:14])
        log(f"fine-tuning smoke (1 epoch on {len(tr[::12])} candidates): held-out scores {np.round(r[1][0], 3)}, "
            f"test-path scores {np.round(r[1][1], 3)}")


# ------------------------------------------------------------------------------------------------ reporting

def table() -> None:
    sp = make_split(load_bundle(signals=False))
    for name in PROBES:
        p = FM / "probe" / f"{name}.json"
        if p.exists():
            res = json.loads(p.read_text())
            print(f"\n## {name} (selected {res['selected']['embedding']}, C={res['selected']['C']:g})\n")
            print("| embedding | C | dev OOF AUC | mean fold AUC |\n|---|---|---|---|")
            for r in res["grid"]:
                print(f"| {r['embedding']} | {r['C']:g} | {r['auc']:.3f} | {r['mean_fold_auc']:.3f} |")
    rows = grid_rows(FM / "ft" / "grid", sp)
    for model in ("jepa", "founder"):
        names = {ft_name(c) for c in ft_configs(model)}
        print(f"\n## {FT_NAMES[model]}\n")
        print(rows_markdown([r for r in rows if r["config"] in names], ["ckpt", "mapping", "lr"]))


PREP_NOTE = {"jepa_probe": "250 Hz mV as in the bundle (notch only), no normalisation (linear_eval.py)",
             "jepa_ft": "250 Hz mV, 0.67 Hz high-pass + 40 Hz low-pass (5th-order Butterworth, zero phase; "
                        "finetuning.py), no normalisation",
             "founder": "linear interpolation to 500 Hz; 50 Hz notch, 0.67-40 Hz band-pass (4th order, filtfilt), "
                        "minus 0.4-s median baseline (util.filter_bandpass); z-score over the segment's real leads "
                        "(dataset.py)"}
MAPPING_NOTE = {"a": "name mapping into the model's lead slots (slotless leads into free slots in a fixed order)",
                "b": "per lead: each real lead alone, embeddings averaged over leads"}


def finish_all() -> None:
    b = load_bundle(signals=False)
    sp = make_split(b)
    dev_ids, test_ids = [b.ids[i] for i in sp.dev], [b.ids[i] for i in sp.test]
    spec = {n: (f, c, m, p) for n, f, c, m, p in EMBEDDINGS}
    for name in ("ecgjepa_probe", "ecgjepa_random_probe", "ecgfounder_probe"):
        p = FM / "probe" / f"{name}.json"
        if not p.exists():
            print(f"{name}: missing")
            continue
        res = json.loads(p.read_text())
        sel = res["selected"]
        family, ckpt, mapping, _ = spec[sel["embedding"]]
        other = {}
        for emb in PROBES[name]:
            best = max((r for r in res["grid"] if r["embedding"] == emb), key=lambda r: r["auc"])
            other[emb] = {"C": best["C"], "dev_oof_auc": best["auc"]}
        model = {"jepa": "ECG-JEPA", "founder": "ECGFounder"}[family]
        ck = {"init": "random initialisation (repository init, seed 0)", "multiblock": "multi-block masking",
              "random": "random masking", "12lead": "12-lead model", "1lead": "single-lead model"}[ckpt]
        notes = (f"{model} ({ck}) frozen, linear probe: features [e_cur, e_ref, e_cur - e_ref] (e_cur = mean over the "
                 f"8 current segments), standardised, L2 logistic regression. Selected by pooled out-of-fold "
                 f"development AUC over {PROBES[name]} x C {list(C_GRID)}: {sel['embedding']}, C={sel['C']:g} "
                 f"(AUC {sel['auc']:.3f}). Lead mapping ({mapping}): {MAPPING_NOTE[mapping]}. Preprocessing: "
                 f"{PREP_NOTE['jepa_probe' if family == 'jepa' else 'founder']}. Development values are out-of-fold; "
                 f"test = mean of the 5 fold models.")
        extra = {"config": {"model": model, "checkpoint": ck, "lead_mapping": mapping, "C": sel["C"],
                            "embedding": sel["embedding"],
                            "preprocessing": PREP_NOTE['jepa_probe' if family == 'jepa' else 'founder']},
                 "best_per_option": other, "grid": res["grid"]}
        r = finish(name, b, dict(zip(dev_ids, res["dev"])), dict(zip(test_ids, res["test"])), notes, extra=extra)
        print(name, {s: round(d["auc"], 3) for s, d in r["splits"].items()})
    for model, name in FT_NAMES.items():
        root = FM / "ft" / "final" / model
        if not (root / f"fold{N_FOLDS - 1}.npz").exists():
            print(f"{name}: final run missing")
            continue
        sel = json.loads((root / "selected.json").read_text())
        best = sel["selected"]
        dev, test = final_scores(root, sp)
        grid = [{k: r[k] for k in ("config", "ckpt", "mapping", "lr", "epochs", "auc", "mean_fold_auc")}
                for r in sel["grid"]]
        mname = {"jepa": "ECG-JEPA", "founder": "ECGFounder"}[model]
        ck = (best["ckpt"] + " masking checkpoint" if model == "jepa"
              else ("12-lead model" if best["mapping"] == "a" else "single-lead model"))
        prep = PREP_NOTE["jepa_ft" if model == "jepa" else "founder"]
        notes = (f"{mname} ({ck}) fully fine-tuned with a linear head on [h_cur, h_ref, h_cur - h_ref] of (current "
                 f"segment, reference segment) pairs; candidate score = mean of the 8 per-segment probabilities. "
                 f"Selected by pooled out-of-fold development AUC over {len(grid)} grid points: mapping "
                 f"{best['mapping']} ({MAPPING_NOTE[best['mapping']]}), lr {best['lr']:g}, {best['epochs']} epochs "
                 f"(grid AUC {best['auc']:.3f}); batch {FT_BATCH}, warm-up {WARMUP} steps then constant, "
                 + ("AdamW betas (0.9, 0.95) wd 0.05" if model == "jepa" else "Adam wd 1e-5")
                 + f". Preprocessing: {prep}. The selected configuration was retrained on each fold (same seeds): "
                   f"development values are out-of-fold from these models, test = mean of the 5 fold models.")
        extra = {"config": {"model": mname, "checkpoint": ck, "lead_mapping": best["mapping"], "lr": best["lr"],
                            "epochs": best["epochs"], "batch": FT_BATCH, "warmup_steps": WARMUP,
                            "optimizer": "AdamW(0.9, 0.95), wd 0.05" if model == "jepa" else "Adam, wd 1e-5",
                            "preprocessing": prep, "seeds": "fold index"},
                 "grid": grid}
        r = finish(name, b, dict(zip(dev_ids, dev.tolist())), dict(zip(test_ids, test.tolist())), notes, extra=extra)
        print(name, {s: round(d["auc"], 3) for s, d in r["splits"].items()})


def finish_paperprep() -> None:
    """Extra row `ecgfounder_probe_paperprep` (added after the pre-specified runs, coordinator decision 2026-10-12):
    the ECGFounder linear probe on the paper's preprocessing (0.5 Hz high-pass, 2nd-order 50 Hz low-pass, 50 Hz
    notch, z-score; no 0.4-s median baseline), with the lead option and C chosen by the development CV of
    `founder_paper_sensitivity`. Refitted locally from the saved embeddings (development out-of-fold recomputed with
    the local scikit-learn and compared with the pod's values); the test set is scored once here. The pre-specified
    repo-pipeline row `ecgfounder_probe` is unchanged."""
    import sklearn
    b = load_bundle(signals=False)
    sp = make_split(b)
    yd = sp.y[sp.dev].astype(bool)
    res = json.loads((FM / "probe" / "founder_paper_sensitivity.json").read_text())
    sel = res["selected"]
    assert sel["embedding"] == "founder_paper_b", sel
    z = dict(np.load(FM / "emb" / f"{sel['embedding']}.npz"))
    assert list(z["ids"]) == b.ids
    X, Xt = probe_features(z, sp.dev), probe_features(z, sp.test)
    oof, tests = np.zeros(len(sp.dev)), []
    for k in range(N_FOLDS):
        m = probe_model(sel["C"]).fit(X[sp.fold != k], yd[sp.fold != k])
        oof[sp.fold == k] = m.predict_proba(X[sp.fold == k])[:, 1]
        tests.append(m.predict_proba(Xt)[:, 1])
    diff = float(np.abs(oof - np.array(res["dev"])).max())
    log(f"local refit (scikit-learn {sklearn.__version__}): dev OOF AUC {auc(oof, yd):.4f} "
        f"(pod {sel['auc']:.4f}); max |OOF difference| {diff:.2e}")
    best_a = max((r for r in res["grid"] if r["embedding"] == "founder_paper_a"), key=lambda r: r["auc"])
    repo = json.loads((FM / "probe" / "ecgfounder_probe.json").read_text())["selected"]
    prep = ("linear interpolation to 500 Hz; 0.5 Hz high-pass and 50 Hz low-pass (2nd-order Butterworth, zero "
            "phase), 50 Hz notch (Q 30, filtfilt); z-score over the segment's real leads (paper text); no 0.4-s "
            "median baseline")
    notes = (f"ECGFounder (single-lead model) frozen, linear probe as ecgfounder_probe but with the preprocessing "
             f"described in the ECGFounder paper instead of the repository's dataset.py. Added after the pre-specified "
             f"runs: the preprocessing variant was compared on development only after the pre-specified repo-pipeline "
             f"run (pooled out-of-fold development AUC: repo pipeline {repo['auc']:.3f}, paper pipeline "
             f"{sel['auc']:.3f}); the repo-pipeline row ecgfounder_probe is kept unchanged. Lead option and C chosen "
             f"by development CV of the paper pipeline: mapping b ({MAPPING_NOTE['b']}), C={sel['C']:g} (option a "
             f"best {best_a['auc']:.3f} at C={best_a['C']:g}). Preprocessing: {prep}. Development values are "
             f"out-of-fold (refitted locally, scikit-learn {sklearn.__version__}); test = mean of the 5 fold models, "
             f"scored once.")
    extra = {"config": {"model": "ECGFounder", "checkpoint": "single-lead model", "lead_mapping": "b",
                        "C": sel["C"], "embedding": sel["embedding"], "preprocessing": prep},
             "added_after_prespecified_runs": {
                 "decision": "coordinator, 2026-10-12: promote the paper-preprocessing probe as an extra row, "
                             "selected on development only",
                 "dev_oof_auc_repo_pipeline": repo["auc"], "dev_oof_auc_paper_pipeline": sel["auc"],
                 "ecgfounder_probe": "kept unchanged (pre-specified repo pipeline)"},
             "grid": res["grid"], "local_refit": {"sklearn": sklearn.__version__, "max_abs_oof_diff_vs_pod": diff}}
    r = finish("ecgfounder_probe_paperprep", b, {b.ids[i]: float(s) for i, s in zip(sp.dev, oof)},
               {b.ids[i]: float(s) for i, s in zip(sp.test, np.mean(tests, axis=0))}, notes, extra=extra)
    print("ecgfounder_probe_paperprep", {s: round(d["auc"], 3) for s, d in r["splits"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "embed", "probe", "ft-grid", "ft-final", "table", "finish",
                                    "finish-paperprep"])
    ap.add_argument("--model", choices=["jepa", "founder"], action="append")
    ap.add_argument("--only", nargs="*", help="embed: only these embeddings")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "check":
        run_check()
    elif a.cmd == "embed":
        run_embed(a.only)
    elif a.cmd == "probe":
        run_probe()
    elif a.cmd == "ft-grid":
        run_ft_grid(a.worker, a.workers, a.model or ["jepa", "founder"])
    elif a.cmd == "ft-final":
        for m in a.model or ["jepa", "founder"]:
            run_ft_final(m)
    elif a.cmd == "table":
        table()
    elif a.cmd == "finish-paperprep":
        finish_paperprep()
    else:
        finish_all()

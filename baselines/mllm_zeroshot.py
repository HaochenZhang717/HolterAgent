"""Baseline B6: zero-shot ECG-specific multimodal LLMs on the rendered ECG images the agent sees.

Question answered: does a domain-specific ECG MLLM, used zero-shot (no fine-tuning, no examples), judge candidate
ST episodes of 24-h Holter recordings better than our agent? Inputs are the fixed images of
outputs/baselines/_bundle (baselines/bundle.py): strip.png (10-s raw strip at the ST extremum), strip_ref.png (the
same at the reference window) and overlay.png (reference vs extremum averaged beats). LTSTDB records have 2-3
ambulatory leads, not 12 leads, which is out of distribution for both models; that is part of what is measured.

Models (served on GPU server by sky/bl_mllm.yaml; this client runs on the laptop through sky/tunnel.sh):
- PULSE-7B (Liu et al. 2024, arXiv 2410.19008; llava-v1.6-vicuna-7b fine-tuned on ECGInstruct), run with the
  authors' LLaVA code (baselines/mllm_pulse_server.py): conversation template llava_v1, anyres images, fp16. Prompts
  use the closed-ended ECG-QA format of ECGInstruct / ECGBench ("Given this ECG image, please answer the following
  question. ... Options: no, yes ... Only answer based on the given Options without any explanation."). PULSE was
  trained on single images and has a 4096-token context (one strip = 2304 image tokens), so the variants with
  several images are given to it as one composite image.
  Score = P(yes) / (P(yes) + P(no)) of the first answer token (yes = "yes"/"Yes", no = "no"/"No"); label = its greedy
  answer.
- ECG-R1-8B-RL (Jin et al. 2026, arXiv 2602.04279; Qwen3-VL-8B with protocol-guided reasoning and RL), image-only
  mode (the ECG-signal tower is not used; vLLM with the Qwen3-VL architecture). System prompt, user-message layout
  (images, then "\\n" and the text) and decoding (greedy, max 2048 new tokens, repetition penalty 1.0) follow the
  authors' swift template and scripts/shells/inference.sh. Turn 1 is its RL training instruction ("Interpret the
  provided ECG image, identify key features and abnormalities in each lead, and generate a clinical diagnosis that
  is supported by the observed evidence."), so that it reasons as trained (<think> Steps 1-6, including "Ischemia,
  Infarction & Mimics" </think>, report, <answer> diagnoses </answer>); turn 2, after its own interpretation, asks
  the closed-ended ECG-QA question in the same format as for PULSE, and one answer token is read with the top-20
  log-probabilities: score = P(yes) / (P(yes) + P(no)), label = the most likely token. (A format pilot on three
  development cases showed that with the yes/no question in the first turn the model answers at once without
  reasoning, hence the two turns.)

Protocol (baselines/learned_common.py): three variants per model, fixed before any result was seen (strip;
reference strip + strip; reference strip + strip + overlay); every variant is run on the development set, the one
with the highest development AUC is chosen (`select`), and only that one is run on the test set, once. The
threshold comes from the development scores (common.evaluate). Raw outputs, one JSON line per candidate, go to
outputs/baselines/<model>/raw/<variant>_<split>.jsonl (resumable); test outputs are never printed.

    python -m baselines.mllm_zeroshot run --model pulse7b --split dev          # all variants
    python -m baselines.mllm_zeroshot select --model pulse7b                   # writes raw/choice.json
    python -m baselines.mllm_zeroshot run --model pulse7b --split test         # the chosen variant only
    python -m baselines.mllm_zeroshot finish --model pulse7b                   # outputs/baselines/pulse7b/
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from baselines.common import OUT, auc, best_threshold, bootstrap, rates
from baselines.learned_common import Bundle, finish, load_bundle

VARIANTS = ("strip", "ref_strip", "ref_strip_overlay")
URL = {"pulse7b": "http://localhost:8011", "ecgr1": "http://localhost:8010/v1"}
WORKERS = {"pulse7b": 2, "ecgr1": 48}

# --- prompts --------------------------------------------------------------------------------------------------
QA_HEAD = ("Given this ECG image, please answer the following question. From the provided options, select all "
           "that apply. List your selections, separated by commas.")
QA_TAIL = "Options: no, yes\n\nOnly answer based on the given Options without any explanation."
ECGR1_SYSTEM = "You are a helpful, harmless clinical ECG assistant. Provide concise, evidence-based interpretations."
HOLTER = "{n}-lead ambulatory (Holter) ECG (leads {leads}), not a standard 12-lead ECG"
CONTEXT = {
    "pulse7b": {
        "strip": "The image is a 10-second strip of a " + HOLTER + ".",
        "ref_strip": ("The image shows two 10-second strips of the same " + HOLTER + ": the top strip is the "
                      "patient's baseline, the bottom strip was recorded during the episode in question."),
        "ref_strip_overlay": ("The image shows, on the left, two 10-second strips of the same " + HOLTER + ": the "
                              "top strip is the patient's baseline, the bottom strip was recorded during the "
                              "episode in question; on the right, the averaged beats of the baseline (grey) and "
                              "of the episode (black) superimposed."),
    },
    "ecgr1": {
        "strip": "The image is a 10-second strip of a " + HOLTER + ".",
        "ref_strip": ("The two images are 10-second strips of the same " + HOLTER + ": the first is the patient's "
                      "baseline, the second was recorded during the episode in question."),
        "ref_strip_overlay": ("The first two images are 10-second strips of the same " + HOLTER + ": the first is "
                              "the patient's baseline, the second was recorded during the episode in question. The "
                              "third image shows the averaged beats of the baseline (grey) and of the episode "
                              "(black) superimposed."),
    },
}
QUESTION = {
    "strip": "Does this ECG show ischemic ST-segment changes?",
    "ref_strip": "Compared with the baseline, does the episode show ischemic ST-segment changes?",
    "ref_strip_overlay": "Compared with the baseline, does the episode show ischemic ST-segment changes?",
}
# ECG-R1 reasoning turn: its RL training instruction (verbatim for one image; "compare the episode with the
# baseline" added when there are several images), then a follow-up turn with the closed-ended ECG-QA question.
ECGR1_TASK = {
    "strip": ("Interpret the provided ECG image, identify key features and abnormalities in each lead, and generate "
              "a clinical diagnosis that is supported by the observed evidence."),
    "ref_strip": ("Interpret the provided ECG images, identify key features and abnormalities in each lead, compare "
                  "the episode with the baseline, and generate a clinical diagnosis that is supported by the "
                  "observed evidence."),
}
ECGR1_TASK["ref_strip_overlay"] = ECGR1_TASK["ref_strip"]
DECODING = {
    "pulse7b": "fp16, greedy (temperature 0, as in the authors' ECGBench evaluation); score from the first answer "
               "token's log-probabilities of one forward pass; greedy answer of at most 8 tokens for the label",
    "ecgr1": "turn 1 (reasoning): greedy (temperature 0), max 2048 new tokens, repetition penalty 1.0 (authors' "
             "inference.sh); turn 2 (ECG-QA question after the model's own interpretation): 1 token, temperature 0, "
             "top-20 logprobs (a most likely whitespace token is appended and the next token read instead)",
}


def prompt_text(model: str, variant: str, leads: list[str]) -> str:
    """PULSE: the question; ECG-R1: the reasoning turn (its question turn is ecgr1_question)."""
    ctx = CONTEXT[model][variant].format(n=len(leads), leads=", ".join(leads))
    if model == "pulse7b":
        return f"{QA_HEAD}\n{ctx}\nQuestion: {QUESTION[variant]}\n{QA_TAIL}"
    # ECG-R1: the text follows the image tokens after a newline, as "<image>\n..." in its training data.
    return f"\n{ctx}\n{ECGR1_TASK[variant]}"


def ecgr1_question(variant: str) -> str:
    head = QA_HEAD if variant == "strip" else QA_HEAD.replace("this ECG image", "these ECG images")
    return f"{head}\nQuestion: {QUESTION[variant]}\n{QA_TAIL}"


# --- images ---------------------------------------------------------------------------------------------------
def _rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def _stack(ims: list[Image.Image], vertical: bool, gap: int = 10) -> Image.Image:
    w = max(i.width for i in ims) if vertical else sum(i.width for i in ims) + gap * (len(ims) - 1)
    h = sum(i.height for i in ims) + gap * (len(ims) - 1) if vertical else max(i.height for i in ims)
    out, pos = Image.new("RGB", (w, h), "white"), 0
    for i in ims:
        if vertical:
            out.paste(i, ((w - i.width) // 2, pos))
            pos += i.height + gap
        else:
            out.paste(i, (pos, (h - i.height) // 2))
            pos += i.width + gap
    return out


def images_for(b: Bundle, cid: str, model: str, variant: str) -> list[Image.Image]:
    kinds = {"strip": ["strip"], "ref_strip": ["strip_ref", "strip"],
             "ref_strip_overlay": ["strip_ref", "strip", "overlay"]}[variant]
    ims = [_rgb(b.image(cid, k)) for k in kinds]
    if model == "ecgr1" or len(ims) == 1:
        return ims
    # PULSE: one composite image (single-image model, 4096-token context).
    strips = _stack(ims[:2], vertical=True)
    return [strips] if len(ims) == 2 else [_stack([strips, ims[2]], vertical=False)]


def _b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --- model calls ----------------------------------------------------------------------------------------------
def _yes_no(text: str) -> bool | None:
    m = re.match(r"\W*(yes|no)\b", text.strip().lower())
    return None if m is None else m.group(1) == "yes"


def _p_yes(logp_yes: float | None, logp_no: float | None) -> float | None:
    if logp_yes is None and logp_no is None:
        return None
    if logp_no is None:
        return 1.0
    if logp_yes is None:
        return 0.0
    return 1.0 / (1.0 + math.exp(logp_no - logp_yes))


def call_pulse(ims: list[Image.Image], text: str, question: str | None = None) -> dict:
    r = requests.post(URL["pulse7b"] + "/score", json={"image": _b64(ims[0]), "question": text,
                                                       "max_new_tokens": 8}, timeout=600)
    out = r.json()
    if r.status_code != 200:
        raise RuntimeError(out.get("error"))
    out["score"] = _p_yes(out["logp_yes"], out["logp_no"])
    out["label"] = _yes_no(out["text"])
    return out


def _chat(messages: list[dict], **kw) -> dict:
    r = requests.post(URL["ecgr1"] + "/chat/completions", json={"model": "ecg-r1", "messages": messages, **kw},
                      timeout=1800)
    r.raise_for_status()
    j = r.json()
    return {**j["choices"][0], "usage": j.get("usage") or {}}


def call_ecgr1(ims: list[Image.Image], text: str, question: str) -> dict:
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(i)}} for i in ims]
    messages = [{"role": "system", "content": ECGR1_SYSTEM},
                {"role": "user", "content": content + [{"type": "text", "text": text}]}]
    gen = _chat(messages, temperature=0.0, max_tokens=2048, repetition_penalty=1.0)
    reasoning = gen["message"]["content"] or ""
    m = re.search(r"<answer>(.*?)(</answer>|$)", reasoning, flags=re.S)
    messages += [{"role": "assistant", "content": reasoning}, {"role": "user", "content": question}]
    prefix = ""
    for _ in range(3):  # if the most likely next token is whitespace, step over it
        ans = _chat(messages + ([{"role": "assistant", "content": prefix}] if prefix else []), temperature=0.0,
                    max_tokens=1, logprobs=True, top_logprobs=20,
                    **({"continue_final_message": True, "add_generation_prompt": False} if prefix else {}))
        top = ans["logprobs"]["content"][0]["top_logprobs"]
        if top[0]["token"].strip():
            break
        prefix += top[0]["token"]
    lp = {"yes": [], "no": []}
    for t in top:
        key = t["token"].strip().lower()
        if key in lp:
            lp[key].append(t["logprob"])
    lse = {k: (float(np.logaddexp.reduce(v)) if v else None) for k, v in lp.items()}
    return {"score": _p_yes(lse["yes"], lse["no"]), "label": _yes_no(top[0]["token"]), "logp_yes": lse["yes"],
            "logp_no": lse["no"], "top": [[t["token"], t["logprob"]] for t in top[:10]], "text": reasoning,
            "answer_dx": m.group(1).strip() if m else None, "finish_reason": gen.get("finish_reason"),
            "n_reasoning_tokens": gen["usage"].get("completion_tokens"),
            "has_think": "</think>" in reasoning, "has_answer": m is not None, "answer_prefix": prefix}


def final_score(r: dict) -> float:
    """P(yes); an answer token without yes/no among its log-probabilities falls back to the model's own label."""
    if r["score"] is not None:
        return float(r["score"])
    return 1.0 if r.get("label") is True else 0.0 if r.get("label") is False else 0.5


# --- runs -----------------------------------------------------------------------------------------------------
def raw_path(model: str, variant: str, split: str) -> Path:
    return OUT / model / "raw" / f"{variant}_{split}.jsonl"


def load_raw(model: str, variant: str, split: str) -> dict[str, dict]:
    p = raw_path(model, variant, split)
    if not p.exists():
        return {}
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return {r["cid"]: r for r in rows if "error" not in r}


def run(model: str, variant: str, split: str, b: Bundle, limit: int | None = None) -> None:
    idx = b.dev if split == "dev" else b.test
    done = load_raw(model, variant, split)
    todo = [b.ids[i] for i in idx if b.ids[i] not in done][:limit]
    print(f"[{model}/{variant}/{split}] {len(done)} done, {len(todo)} to run", flush=True)
    path = raw_path(model, variant, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock, call = threading.Lock(), (call_pulse if model == "pulse7b" else call_ecgr1)

    def one(cid: str) -> bool:
        leads = b.meta[cid]["leads"]
        text = prompt_text(model, variant, leads)
        row = None
        for attempt in range(20):  # a dropped tunnel to the server is waited out (2026-10-23), other errors recorded
            try:
                row = {"cid": cid, "split": split, "variant": variant,
                       **call(images_for(b, cid, model, variant), text, ecgr1_question(variant))}
                break
            except requests.ConnectionError as e:
                row = {"cid": cid, "split": split, "variant": variant, "error": f"{type(e).__name__}: {e}"[:500]}
                time.sleep(30)
            except Exception as e:  # noqa: BLE001 - recorded; the case is retried on the next run
                row = {"cid": cid, "split": split, "variant": variant, "error": f"{type(e).__name__}: {e}"[:500]}
                break
        with lock, path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return "error" not in row

    n_ok = n_err = 0
    with ThreadPoolExecutor(WORKERS[model]) as ex:
        for k, fut in enumerate(as_completed([ex.submit(one, c) for c in todo]), 1):
            n_ok, n_err = n_ok + fut.result(), n_err + (not fut.result())
            if k % 50 == 0 or k == len(todo):
                print(f"[{model}/{variant}/{split}] {k}/{len(todo)} (errors {n_err})", flush=True)


def dev_metrics(b: Bundle, rows: dict[str, dict]) -> dict:
    """Development AUC (record-bootstrap 95% CI) and rates at the development threshold, as common.evaluate."""
    ids = [b.ids[i] for i in b.dev]
    y = b.y[b.dev]
    recs = list(b.record[b.dev])
    s = np.array([final_score(rows[c]) for c in ids], float)
    thr = best_threshold(s, y)
    lab = [rows[c].get("label") for c in ids]
    return {"n": len(ids), "n_without_yes_no_logprob": sum(rows[c]["score"] is None for c in ids), "auc": auc(s, y),
            "auc_ci": bootstrap(lambda i: auc(s[i], y[i]), recs), "threshold": thr,
            "at_dev_threshold": rates(s >= thr, y),
            "own_label": rates(np.array([v is True for v in lab]), y),
            "n_unparsable_label": int(sum(v is None for v in lab))}


def summary_counts(rows: dict[str, dict]) -> dict:
    lab = [r.get("label") for r in rows.values()]
    out = {"n": len(rows), "label_yes": sum(v is True for v in lab), "label_no": sum(v is False for v in lab),
           "label_unparsable": sum(v is None for v in lab),
           "score_missing": sum(r.get("score") is None for r in rows.values())}
    if any("finish_reason" in r for r in rows.values()):
        out["reasoning_truncated"] = sum(r.get("finish_reason") == "length" for r in rows.values())
        out["no_think_block"] = sum(not r.get("has_think") for r in rows.values())
        out["no_answer_tag"] = sum(not r.get("has_answer") for r in rows.values())
    return out


def select(model: str, b: Bundle) -> dict:
    res = {}
    for v in VARIANTS:
        rows = load_raw(model, v, "dev")
        if len(rows) < len(b.dev):
            raise SystemExit(f"{model}/{v}: {len(rows)}/{len(b.dev)} development cases; run them first")
        res[v] = {**dev_metrics(b, rows), "counts": summary_counts(rows)}
        print(f"[{model}] dev {v}: AUC {res[v]['auc']:.3f} "
              f"({res[v]['auc_ci'][0]:.3f}-{res[v]['auc_ci'][1]:.3f}), {res[v]['counts']}")
    chosen = max(VARIANTS, key=lambda v: res[v]["auc"])
    choice = {"model": model, "chosen": chosen, "rule": "highest development AUC", "dev": res}
    path = OUT / model / "raw" / "choice.json"
    if path.exists() and json.loads(path.read_text())["chosen"] != chosen:
        raise SystemExit(f"{path} already chose another variant")
    path.write_text(json.dumps(choice, indent=1))
    print(f"[{model}] chosen variant: {chosen}")
    return choice


def finish_model(model: str, b: Bundle) -> dict:
    choice = json.loads((OUT / model / "raw" / "choice.json").read_text())
    v = choice["chosen"]
    dev, test = load_raw(model, v, "dev"), load_raw(model, v, "test")
    assert len(dev) == len(b.dev) and len(test) == len(b.test), "development or test run incomplete"
    rows = dev | test

    dev_score = {c: final_score(r) for c, r in dev.items()}
    test_score = {c: final_score(r) for c, r in test.items()}
    label = {c: bool(r.get("label")) for c, r in rows.items()}
    leads = b.meta[b.ids[0]]["leads"]
    # common.evaluate overwrites splits[*].f1_ci with the CI of the own-label F1 when `label` is given; keep the CI
    # of the F1 at the development threshold (what the score-only baselines report) here as well.
    thr = best_threshold(np.array([dev_score[b.ids[i]] for i in b.dev]), b.y[b.dev])
    f1_ci_thr = {}
    for split, idx, sc in (("dev", b.dev, dev_score), ("test", b.test, test_score)):
        sv, yv = np.array([sc[b.ids[i]] for i in idx]), b.y[idx]
        f1_ci_thr[split] = bootstrap(lambda i: rates(sv[i] >= thr, yv[i])["f1"], list(b.record[idx]))
    extra = {
        "f1_ci_at_dev_threshold": f1_ci_thr,
        "f1_ci_note": "splits[*].f1_ci is the CI of the own-label F1 (common.evaluate with label); "
                      "f1_ci_at_dev_threshold is the CI of splits[*].at_dev_threshold.f1",
        "model": {"pulse7b": "PULSE-ECG/PULSE-7B", "ecgr1": "PKUDigitalHealth/ECG-R1-8B-RL"}[model],
        "chosen_variant": v, "selection_rule": choice["rule"],
        "prompts": {var: prompt_text(model, var, ["<lead 1>", "<lead 2>"]) if model == "pulse7b" else
                    {"turn1_after_images": prompt_text(model, var, ["<lead 1>", "<lead 2>"]),
                     "turn2": ecgr1_question(var)} for var in VARIANTS},
        "system_prompt": ECGR1_SYSTEM if model == "ecgr1" else
        "llava_v1: A chat between a curious human and an artificial intelligence assistant. The assistant gives "
        "helpful, detailed, and polite answers to the human's questions.",
        "decoding": DECODING[model],
        "score": "P(yes) / (P(yes) + P(no)) of the answer token; if neither is among the returned log-probabilities, "
                 "the model's own label (1/0), else 0.5",
        "label": "the model's own yes/no answer (unparsable counts as no)",
        "dev_variants_development_only": choice["dev"],
        "counts_chosen": {"dev": summary_counts(dev), "test": summary_counts(test)},
        "example_leads": leads,
    }
    notes = (f"Zero-shot {extra['model']} on the rendered ECG images (variant {v}, chosen by development AUC among "
             f"{', '.join(VARIANTS)}); no training; threshold from the development scores.")
    name = model
    r = finish(name, b, dev_score, test_score, notes, extra=extra, label=label)
    for split in ("dev", "test"):
        s = r["splits"][split]
        print(f"[{name}] {split}: AUC {s['auc']:.3f} ({s['auc_ci'][0]:.3f}-{s['auc_ci'][1]:.3f}); "
              f"at dev threshold {s['at_dev_threshold']}; own label {s.get('rule')}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "select", "finish", "show-prompts"])
    ap.add_argument("--model", choices=list(URL), required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--variant", choices=VARIANTS, help="dev only; default all variants")
    ap.add_argument("--limit", type=int, help="run at most this many cases (smoke test, dev only)")
    a = ap.parse_args()
    b = load_bundle(signals=False)
    if a.cmd == "show-prompts":
        for v in VARIANTS:
            print(f"--- {a.model} / {v}\n{prompt_text(a.model, v, b.meta[b.ids[0]]['leads'])}\n")
    elif a.cmd == "run" and a.split == "dev":
        for v in [a.variant] if a.variant else VARIANTS:
            run(a.model, v, "dev", b, a.limit)
    elif a.cmd == "run":
        if a.variant or a.limit:
            raise SystemExit("the test set runs the chosen variant, complete, once")
        chosen = json.loads((OUT / a.model / "raw" / "choice.json").read_text())["chosen"]
        run(a.model, chosen, "test", b)
    elif a.cmd == "select":
        select(a.model, b)
    else:
        finish_model(a.model, b)


if __name__ == "__main__":
    main()

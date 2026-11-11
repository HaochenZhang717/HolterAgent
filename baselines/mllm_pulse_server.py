"""Minimal HTTP server for PULSE-7B (zero-shot ECG MLLM baseline B6; the client is baselines/mllm_zeroshot.py).

PULSE-7B (Liu et al., "Teach Multimodal LLMs to Comprehend Electrocardiographic Images", arXiv 2410.19008;
https://huggingface.co/PULSE-ECG/PULSE-7B) is llava-v1.6-vicuna-7b fine-tuned on ECGInstruct. Its checkpoint is in
the original LLaVA format (LlavaLlamaForCausalLM, anyres images, mm_patch_merge_type "flat"), which vLLM's
LLaVA-NeXT implementation does not reproduce (it always uses spatial_unpad + image_newline), so this server runs the
authors' own inference code (github.com/AIMedLab/PULSE, branch dev, LLaVA/llava: load_pretrained_model,
process_images, tokenizer_image_token, conversation template "llava_v1", fp16) exactly as in
LLaVA/llava/eval/model_ecg.py, with one change: the CLIP ViT-L/14-336 vision tower is built from its configuration
instead of being downloaded from openai/clip-vit-large-patch14-336, because the PULSE checkpoint contains all 391
fine-tuned vision-tower tensors (unfreeze_mm_vision_tower: true) and they overwrite the downloaded ones anyway.

POST /score {"image": <base64 PNG>, "question": <text after the image token>, "max_new_tokens": 8}
returns the next-token log-probabilities of the answer (yes / no variants), the top-10 next tokens, the greedy
answer text and the exact prompt.

    ~/pulse-env/bin/python baselines/mllm_pulse_server.py --model ~/models/PULSE-7B --port 8011
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.multimodal_encoder import clip_encoder

# openai/clip-vit-large-patch14-336: config.json (vision_config) and preprocessor_config.json, verbatim values.
CLIP_VISION = dict(hidden_size=1024, image_size=336, intermediate_size=4096, num_attention_heads=16,
                   num_hidden_layers=24, patch_size=14, projection_dim=768, hidden_act="quick_gelu",
                   layer_norm_eps=1e-5, num_channels=3, attention_dropout=0.0, dropout=0.0)
CLIP_PREPROCESS = dict(crop_size=336, size=336, do_center_crop=True, do_normalize=True, do_resize=True,
                       image_mean=[0.48145466, 0.4578275, 0.40821073],
                       image_std=[0.26862954, 0.26130258, 0.27577711], resample=3)
YES, NO = ("▁yes", "▁Yes", "yes", "Yes"), ("▁no", "▁No", "no", "No")


def _load_from_config(self, device_map=None):
    """CLIPVisionTower.load_model without the download: the weights come from the PULSE checkpoint."""
    if self.is_loaded:
        return
    self.image_processor = CLIPImageProcessor(**CLIP_PREPROCESS)
    self.vision_tower = CLIPVisionModel(CLIPVisionConfig(**CLIP_VISION))
    self.vision_tower.requires_grad_(False)
    self.is_loaded = True


clip_encoder.CLIPVisionTower.load_model = _load_from_config


# PULSE_DEVICE=cpu (external evaluation, 2026-10-23: GPU server GPU pods were preempted within minutes): the same code on
# the CPU in bfloat16 (fp16 matmuls are slow on CPUs). Default: the GPU in fp16, as before.
DEVICE = os.environ.get("PULSE_DEVICE", "cuda")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.bfloat16


class Pulse:
    def __init__(self, model_path: str):
        name = get_model_name_from_path(model_path)
        assert "pulse" in name.lower(), "the LLaVA builder recognises PULSE by its folder name"
        if DEVICE == "cuda":
            self.tok, self.model, self.proc, _ = load_pretrained_model(model_path, None, name)
        else:
            # device != "cuda" makes the LLaVA builder load with device_map {"": device}
            self.tok, self.model, self.proc, _ = load_pretrained_model(model_path, None, name, device="cpu")
            self.model.to(dtype=DTYPE)
        self.model.eval()
        vt = self.model.get_vision_tower().vision_tower
        assert all(p.device.type == DEVICE for p in vt.parameters()), "vision tower weights missing from checkpoint"
        vocab = self.tok.get_vocab()
        self.yes = sorted({vocab[t] for t in YES if t in vocab})
        self.no = sorted({vocab[t] for t in NO if t in vocab})
        self.lock = threading.Lock()
        print(f"[pulse] loaded {model_path}; yes ids {self.yes}, no ids {self.no}; "
              f"merge={self.model.config.mm_patch_merge_type} aspect={self.model.config.image_aspect_ratio}",
              flush=True)

    @torch.inference_mode()
    def score(self, image: Image.Image, question: str, max_new_tokens: int = 8) -> dict:
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token(prompt, self.tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
        img = process_images([image], self.proc, self.model.config)[0]
        images = img.unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
        with self.lock:
            logits = self.model(input_ids=ids, images=images, image_sizes=[image.size]).logits[0, -1].float()
            out = self.model.generate(ids, images=images, image_sizes=[image.size], do_sample=False,
                                      temperature=0.0, top_p=None, num_beams=1, max_new_tokens=max_new_tokens,
                                      use_cache=True)
        logp = torch.log_softmax(logits, -1)
        top = torch.topk(logp, 10)
        return {
            "logp_yes": float(torch.logsumexp(logp[self.yes], 0)),
            "logp_no": float(torch.logsumexp(logp[self.no], 0)),
            "top": [[self.tok.convert_ids_to_tokens(int(i)), float(v)] for v, i in zip(top.values, top.indices)],
            "text": self.tok.batch_decode(out, skip_special_tokens=True)[0].strip(),
            "n_crops": int(img.shape[0]),
            "prompt": prompt,
        }


def serve(model: Pulse, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - health check
            self._send(200, {"model": "PULSE-7B"})

        def do_POST(self):  # noqa: N802
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                image = Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB")
                self._send(200, model.score(image, req["question"], int(req.get("max_new_tokens", 8))))
            except Exception as e:  # noqa: BLE001 - report to the client, keep serving
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):  # quiet
            pass

    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8011)
    a = ap.parse_args()
    serve(Pulse(a.model), a.port)

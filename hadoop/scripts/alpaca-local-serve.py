#!/usr/bin/env python3
"""HTTP serve для локальной HF-папки (если в Registry нет MLmodel)."""
from __future__ import annotations

import os

from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_DIR = os.environ.get("ALPACA_LOCAL_MODEL_DIR", "/models/alpaca-distilgpt2")
PORT = int(os.environ.get("MLFLOW_SERVE_PORT", "5002"))

app = Flask(__name__)
_trust = os.environ.get("HF_TRUST_REMOTE_CODE", "1") not in ("0", "false", "False")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=_trust)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=_trust)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


@app.get("/health")
def health():
    return "OK", 200


def _trim_alpaca_completion(text: str) -> str:
    """Как в alpaca_llm_utils.generate_sample — обрезка следующего блока Alpaca."""
    for stop in ("\n\n###", "\n### Instruction", "\n### Input"):
        if stop in text:
            text = text.split(stop, 1)[0].rstrip()
    return text


@app.post("/invocations")
def invocations():
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("inputs", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    params = data.get("parameters") or data.get("params") or {}
    max_new = int(params.get("max_new_tokens", 80))
    do_sample = bool(params.get("do_sample", False))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kw = {
        "max_new_tokens": max_new,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": float(params.get("repetition_penalty", 1.15)),
        "no_repeat_ngram_size": int(params.get("no_repeat_ngram_size", 3)),
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kw.update(temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :].lstrip()
    return jsonify([_trim_alpaca_completion(text)])


if __name__ == "__main__":
    print(f"Alpaca local serve: {MODEL_DIR} on :{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)

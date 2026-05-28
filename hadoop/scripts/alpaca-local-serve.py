#!/usr/bin/env python3
"""HTTP serve для локальной HF-папки (если в Registry нет MLmodel)."""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_DIR = os.environ.get("ALPACA_LOCAL_MODEL_DIR", "/models/alpaca-distilgpt2")
PORT = int(os.environ.get("MLFLOW_SERVE_PORT", "5002"))

app = Flask(__name__)
_trust = os.environ.get("HF_TRUST_REMOTE_CODE", "1") not in ("0", "false", "False")


def _resolve_model_dir(model_dir: str) -> str:
    path = Path(model_dir)
    merged = path / "merged"
    if (merged / "config.json").is_file():
        return str(merged)

    def _valid_adapter(directory: Path) -> bool:
        adapter = directory / "adapter_model.safetensors"
        return adapter.is_file() and adapter.stat().st_size >= 1024

    if _valid_adapter(path):
        return str(path)

    checkpoints = sorted(
        (p for p in path.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.partition("-")[2] or "0"),
    )
    for ckpt in reversed(checkpoints):
        if _valid_adapter(ckpt):
            return str(ckpt)
    return str(path)


MODEL_DIR = _resolve_model_dir(MODEL_DIR)
print(f"Resolved model dir: {MODEL_DIR}", flush=True)

import sys

sys.path.insert(0, "/scripts")
from alpaca_prompt_utils import (  # noqa: E402
    apply_safe_deepseek_chat_template,
    build_plain_alpaca_prompt,
    decode_new_tokens,
    deepseek_serve_use_plain,
    is_deepseek_model,
    is_refusal_response,
    parse_alpaca_prompt,
    tokenize_generation_prompt,
    trim_alpaca_completion,
)

print("Loading tokenizer...", flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=_trust, local_files_only=True
    )
except OSError:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=_trust)
print("Tokenizer loaded.", flush=True)
BASE_MODEL_NAME = MODEL_DIR
if os.path.isfile(os.path.join(MODEL_DIR, "adapter_config.json")):
    from peft import PeftConfig, PeftModel

    print("Loading PEFT config...", flush=True)
    peft_cfg = PeftConfig.from_pretrained(MODEL_DIR)
    BASE_MODEL_NAME = peft_cfg.base_model_name_or_path
    print(f"Loading base model: {BASE_MODEL_NAME}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=_trust
    )
    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
else:
    print("Loading full HF model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=_trust)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if is_deepseek_model(BASE_MODEL_NAME):
    apply_safe_deepseek_chat_template(tokenizer)
    mode = "plain Alpaca" if deepseek_serve_use_plain() else "safe chat"
    print(f"DeepSeek serve encoding: {mode}", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Moving model to {device}...", flush=True)
model.to(device)
model.eval()
print(f"Model ready (base={BASE_MODEL_NAME}).", flush=True)


def _generate_once(prompt: str, max_new: int, do_sample: bool) -> tuple[str, str]:
    inputs, enc_mode = tokenize_generation_prompt(
        tokenizer, prompt, base_model_name=BASE_MODEL_NAME
    )
    if os.environ.get("DEEPSEEK_SERVE_DEBUG", "").lower() in ("1", "true", "yes"):
        preview = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)[:900]
        print(f"DEBUG prompt ({enc_mode}):\n{preview}\n---", flush=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kw = {
        "max_new_tokens": max_new,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kw.update(temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    text = decode_new_tokens(tokenizer, out[0], inputs, mode=enc_mode, prompt=prompt)
    return trim_alpaca_completion(text), enc_mode


if is_deepseek_model(BASE_MODEL_NAME):
    _smoke = build_plain_alpaca_prompt("Give three tips for staying healthy.", "")
    try:
        _smoke_out, _smoke_mode = _generate_once(_smoke, 48, False)
        print(f"Startup smoke ({_smoke_mode}): {_smoke_out[:160]!r}", flush=True)
        if is_refusal_response(_smoke_out):
            print(
                "WARN: smoke test looks like refusal — проверьте LoRA и MODEL_DIR.",
                flush=True,
            )
    except Exception as exc:
        print(f"Startup smoke skipped: {exc}", flush=True)


@app.get("/health")
def health():
    return "OK", 200


@app.post("/invocations")
def invocations():
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("inputs", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    params = data.get("parameters") or data.get("params") or {}
    max_new = int(params.get("max_new_tokens", 80))
    do_sample = bool(params.get("do_sample", False))

    text, enc_mode = _generate_once(prompt, max_new, do_sample)
    if is_refusal_response(text) and is_deepseek_model(BASE_MODEL_NAME):
        instruction, input_text = parse_alpaca_prompt(prompt)
        plain_prompt = build_plain_alpaca_prompt(instruction, input_text)
        if deepseek_serve_use_plain():
            print("Refusal on plain — retry with safe chat.", flush=True)
            prev = os.environ.get("DEEPSEEK_SERVE_PLAIN")
            os.environ["DEEPSEEK_SERVE_PLAIN"] = "0"
            try:
                text, enc_mode = _generate_once(prompt, max_new, do_sample)
            finally:
                if prev is None:
                    os.environ.pop("DEEPSEEK_SERVE_PLAIN", None)
                else:
                    os.environ["DEEPSEEK_SERVE_PLAIN"] = prev
        else:
            print("Refusal on chat — retry with plain Alpaca.", flush=True)
            text, enc_mode = _generate_once(plain_prompt, max_new, do_sample)
    return jsonify([text])


if __name__ == "__main__":
    print(f"Alpaca local serve: {MODEL_DIR} on :{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)

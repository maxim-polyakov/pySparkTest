"""Alpaca: EDA и fine-tuning без pandas (PySpark + HuggingFace)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

TEXT_COLS = ("instruction", "input", "output", "text")

ALPACA_PREAMBLE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
)

# System для DeepSeek chat template (train + serve); без «только программирование».
DEEPSEEK_ALPACA_SYSTEM = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Answer helpfully; do not refuse "
    "general knowledge or lifestyle questions."
)

SAFE_DEEPSEEK_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}"
    "{% endif %}{{ bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{ message['content'] }}\n\n"
    "{%- elif message['role'] == 'user' %}### Instruction:\n{{ message['content'] }}\n\n"
    "{%- elif message['role'] == 'assistant' %}### Response:\n{{ message['content'] }}\n"
    "<|EOT|>\n"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}### Response:\n\n{%- endif %}"
)


def apply_safe_deepseek_chat_template(tokenizer) -> None:
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = SAFE_DEEPSEEK_CHAT_TEMPLATE


def _is_deepseek_model(model_name: str | None) -> bool:
    return bool(model_name and "deepseek" in model_name.lower())


def _parse_alpaca_prefix(prefix: str) -> tuple[str, str]:
    """instruction/input из префикса Alpaca (до ### Response:)."""
    import re

    body = prefix.strip()
    if "Below is an instruction" in body and "### Instruction:" in body:
        body = "### Instruction:" + body.split("### Instruction:", 1)[1]
    inst_m = re.search(
        r"### Instruction:\s*\n(.*?)(?=\n### Input:|\n### Response:|\Z)",
        body,
        re.DOTALL,
    )
    in_m = re.search(
        r"### Input:\s*\n(.*?)(?=\n### Response:|\Z)", body, re.DOTALL
    )
    instruction = inst_m.group(1).strip() if inst_m else ""
    input_text = in_m.group(1).strip() if in_m else ""
    return instruction, input_text


def alpaca_text_to_chat_messages(text: str) -> list[dict[str, str]]:
    """Полный пример Alpaca → messages для apply_chat_template (DeepSeek)."""
    text = text.strip()
    response = ""
    if "### Response:\n" in text:
        prefix, response = text.split("### Response:\n", 1)
        response = response.strip()
    else:
        prefix = text
    instruction, input_text = _parse_alpaca_prefix(prefix)
    user = instruction
    if input_text:
        user += f"\n\n### Input:\n{input_text}"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": DEEPSEEK_ALPACA_SYSTEM},
        {"role": "user", "content": user},
    ]
    if response:
        messages.append({"role": "assistant", "content": response})
    return messages

DEFAULT_DISTILGPT2_MODEL = "distilgpt2"
DEFAULT_DISTILGPT2_OUTPUT_DIR = "/home/jovyan/work/models/alpaca-distilgpt2"

# Компактный DeepSeek для fine-tune на Alpaca в Docker (~6–8 GB RAM с LoRA)
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
DEFAULT_DEEPSEEK_OUTPUT_DIR = "/home/jovyan/work/models/alpaca-deepseek"

# Имена в MLflow Registry (дублируем здесь — не зависим от версии mlflow_utils в kernel)
ALPACA_REGISTERED_MODEL_NAME = "alpaca-causal-lm"
DISTILGPT_REGISTERED_MODEL_NAME = "distilgpt-causal-lm"
DEEPSEEK_REGISTERED_MODEL_NAME = "deepseek-causal-lm"


def _reload_mlflow_utils():
    """Свежий mlflow_utils после правок на диске (Jupyter кэширует старый модуль)."""
    import importlib
    import sys

    if "mlflow_utils" in sys.modules:
        return importlib.reload(sys.modules["mlflow_utils"])
    import mlflow_utils

    return importlib.reload(mlflow_utils)


@dataclass(frozen=True)
class CausalLmTrainProfile:
    """Профиль обучения одной causal LM на Alpaca."""

    label: str
    model_name: str
    output_dir: str
    register_model_name: str
    max_steps: int = 500
    batch_size: int = 1
    learning_rate: float = 2e-4
    max_length: int | None = 512
    use_lora: bool = True
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    cpu_threads: int | None = None


def default_train_profiles(
    *,
    max_steps: int = 500,
    max_length_distil: int | None = None,
    max_length_deepseek: int = 512,
) -> list[CausalLmTrainProfile]:
    """distilgpt2 (alpaca-causal-lm, :5002) + DeepSeek (deepseek-causal-lm, :5003)."""
    return [
        CausalLmTrainProfile(
            label="distilgpt2",
            model_name=DEFAULT_DISTILGPT2_MODEL,
            output_dir=DEFAULT_DISTILGPT2_OUTPUT_DIR,
            register_model_name=DISTILGPT_REGISTERED_MODEL_NAME,
            max_steps=max_steps,
            batch_size=2,
            learning_rate=5e-5,
            max_length=max_length_distil,
            use_lora=False,
            gradient_checkpointing=False,
            trust_remote_code=False,
            cpu_threads=4,
        ),
        CausalLmTrainProfile(
            label="deepseek",
            model_name=DEFAULT_DEEPSEEK_MODEL,
            output_dir=DEFAULT_DEEPSEEK_OUTPUT_DIR,
            register_model_name=DEEPSEEK_REGISTERED_MODEL_NAME,
            max_steps=max_steps,
            batch_size=1,
            learning_rate=2e-4,
            max_length=max_length_deepseek,
            use_lora=True,
            gradient_checkpointing=True,
            trust_remote_code=True,
            cpu_threads=4,
        ),
    ]


def configure_cpu_threads(cpu_threads: int | None = None) -> None:
    """Настроить параллелизм CPU для PyTorch/BLAS перед обучением."""
    if not cpu_threads or cpu_threads < 1:
        return

    threads = str(cpu_threads)
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads

    try:
        import torch

        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, cpu_threads // 2)))
    except RuntimeError:
        # interop threads нельзя менять после старта параллельной работы;
        # основные torch threads всё равно могли примениться.
        pass


def build_alpaca_prompt(
    instruction: str,
    *,
    input_text: str = "",
    with_response_header: bool = True,
) -> str:
    """Шаблон Alpaca (instruction + опциональный input + заголовок Response)."""
    parts = [ALPACA_PREAMBLE, f"### Instruction:\n{instruction.strip()}\n\n"]
    if str(input_text).strip():
        parts.append(f"### Input:\n{str(input_text).strip()}\n\n")
    if with_response_header:
        parts.append("### Response:\n")
    return "".join(parts)


def prompt_for_generation_from_row(row) -> str:
    """Промпт как при обучении: из колонки text или из instruction/input."""
    text = (getattr(row, "text", None) or "").strip()
    marker = "### Response:\n"
    if text and marker in text:
        return text.split(marker, 1)[0] + marker
    return build_alpaca_prompt(
        getattr(row, "instruction", "") or "",
        input_text=getattr(row, "input", "") or "",
    )


def reference_from_alpaca_text(text: str) -> str:
    """Эталонный output из поля text датасета Alpaca."""
    marker = "### Response:\n"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return ""


def collect_eval_prompt_pairs(
    eval_df: DataFrame,
    *,
    max_samples: int = 20,
    text_column: str = "text",
) -> list[tuple[str, str]]:
    """Пары (prompt, reference) из eval Spark DataFrame для метрик serve."""
    from types import SimpleNamespace

    pairs: list[tuple[str, str]] = []
    rows = eval_df.select(text_column).limit(max_samples).collect()
    for row in rows:
        text = (getattr(row, text_column, None) or "").strip()
        if not text:
            continue
        ref = reference_from_alpaca_text(text)
        if not ref:
            continue
        prompt = prompt_for_generation_from_row(
            SimpleNamespace(text=text, instruction="", input="", output="")
        )
        pairs.append((prompt, ref))
    return pairs


def with_text_lengths(sdf: DataFrame) -> DataFrame:
    out = sdf
    for col in TEXT_COLS:
        c = F.coalesce(F.col(col), F.lit(""))
        out = (
            out.withColumn(f"{col}_len", F.length(c))
            .withColumn(f"{col}_words", F.size(F.split(F.trim(c), r"\s+")))
        )
    return out.withColumn(
        "has_input",
        F.length(F.trim(F.coalesce(F.col("input"), F.lit("")))) > 0,
    )


def eda_summary_spark(sdf: DataFrame) -> list[tuple[str, Any]]:
    df = with_text_lengths(sdf)
    total = df.count()
    empty_input = df.filter(~F.col("has_input")).count()
    dup_groups = (
        df.groupBy("instruction").count().filter(F.col("count") > 1).count()
    )
    row = df.agg(
        F.expr("percentile_approx(instruction_len, 0.5)").alias("instruction_med"),
        F.expr("percentile_approx(output_len, 0.5)").alias("output_med"),
        F.expr("percentile_approx(text_len, 0.5)").alias("text_med"),
        F.expr("percentile_approx(output_len, 0.95)").alias("output_p95"),
    ).collect()[0]
    return [
        ("rows", total),
        ("instruction_with_duplicates", dup_groups),
        ("empty_input_pct", round(100 * empty_input / total, 2) if total else 0),
        ("instruction_len_median", int(row.instruction_med or 0)),
        ("output_len_median", int(row.output_med or 0)),
        ("text_len_median", int(row.text_med or 0)),
        ("output_len_p95", int(row.output_p95 or 0)),
    ]


def length_histogram(
    sdf: DataFrame,
    len_col: str,
    *,
    bin_width: int = 50,
) -> tuple[list[float], list[int]]:
    p99 = sdf.approxQuantile(len_col, [0.99], 0.01)[0]
    rows = (
        sdf.filter(F.col(len_col) <= p99)
        .withColumn("bin", (F.floor(F.col(len_col) / bin_width) * bin_width).cast("int"))
        .groupBy("bin")
        .agg(F.count("*").alias("cnt"))
        .orderBy("bin")
        .collect()
    )
    return [float(r.bin) for r in rows], [int(r.cnt) for r in rows]


def top_instruction_words(sdf: DataFrame, top_n: int = 25) -> DataFrame:
    return (
        sdf.select(
            F.explode(
                F.split(
                    F.lower(F.trim(F.coalesce(F.col("instruction"), F.lit("")))),
                    r"\s+",
                )
            ).alias("word")
        )
        .filter(F.col("word") != "")
        .groupBy("word")
        .agg(F.count("*").alias("cnt"))
        .orderBy(F.desc("cnt"))
        .limit(top_n)
    )


def top_instruction_words_lists(
    sdf: DataFrame, top_n: int = 25
) -> tuple[list[str], list[int]]:
    """Слова и частоты для matplotlib (без Row.count)."""
    rows = top_instruction_words(sdf, top_n).collect()
    return [r.word for r in rows], [int(r.cnt) for r in rows]


def build_train_eval_spark(
    sdf: DataFrame,
    *,
    max_samples: int | None = 5000,
    test_frac: float = 0.1,
    seed: int = 42,
    text_column: str = "text",
) -> tuple[DataFrame, DataFrame]:
    base = sdf.select(text_column).filter(
        F.col(text_column).isNotNull() & (F.length(F.trim(F.col(text_column))) > 0)
    )
    if max_samples:
        base = base.orderBy(F.rand(seed)).limit(max_samples)

    split = base.withColumn("_r", F.rand(seed + 1))
    eval_df = split.filter(F.col("_r") < test_frac).select(text_column)
    train_df = split.filter(F.col("_r") >= test_frac).select(text_column)
    return train_df, eval_df


def spark_texts_to_tokenized(
    train_df: DataFrame,
    eval_df: DataFrame,
    tokenizer,
    text_column: str = "text",
    *,
    max_length: int | None = None,
    model_name: str | None = None,
):
    """Spark → torch Dataset для Trainer (без huggingface `datasets`/pyarrow).

    max_length=None — truncation до model_max_length токенизатора, padding в collator.
    """
    from torch.utils.data import Dataset as TorchDataset

    train_texts = [r[text_column] for r in train_df.collect()]
    eval_texts = [r[text_column] for r in eval_df.collect()]

    if not model_name:
        model_name = getattr(tokenizer, "name_or_path", None) or ""

    use_chat = _is_deepseek_model(model_name) and hasattr(
        tokenizer, "apply_chat_template"
    )
    if use_chat:
        apply_safe_deepseek_chat_template(tokenizer)

    def _encode(texts: list[str]):
        if use_chat:
            cap = max_length
            if cap is None:
                cap = getattr(tokenizer, "model_max_length", None) or 1024
                if cap > 100_000:
                    cap = 1024
            ids_list: list[list[int]] = []
            mask_list: list[list[int]] = []
            for text in texts:
                messages = alpaca_text_to_chat_messages(text)
                row_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=cap,
                )
                if isinstance(row_ids, dict):
                    ids_list.append(list(row_ids["input_ids"]))
                    mask_list.append(
                        list(row_ids.get("attention_mask") or [1] * len(row_ids["input_ids"]))
                    )
                else:
                    ids_list.append(list(row_ids))
                    mask_list.append([1] * len(row_ids))
            if max_length is not None:
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                for i, seq in enumerate(ids_list):
                    if len(seq) < max_length:
                        pad_len = max_length - len(seq)
                        ids_list[i] = seq + [pad_id] * pad_len
                        mask_list[i] = mask_list[i] + [0] * pad_len
            return {"input_ids": ids_list, "attention_mask": mask_list}

        if max_length is None:
            cap = getattr(tokenizer, "model_max_length", None) or 1024
            if cap > 100_000:
                cap = 1024
            return tokenizer(
                texts,
                truncation=True,
                max_length=cap,
                padding=False,
            )
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    class _Encoded(TorchDataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    return _Encoded(_encode(train_texts)), _Encoded(_encode(eval_texts))


def _load_causal_lm(
    model_name: str,
    *,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM

    load_kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if not torch.cuda.is_available():
        load_kw["low_cpu_mem_usage"] = True
    if torch.cuda.is_available():
        load_kw["torch_dtype"] = torch.float16
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except ImportError:
            pass

    print(f"model: loading {model_name} with {load_kw}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    print(f"model: loaded {model.__class__.__name__}", flush=True)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("model: gradient checkpointing enabled", flush=True)

    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "Для LoRA установите peft: pip install 'peft>=0.11.0'"
            ) from exc
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        print("model: LoRA adapter attached", flush=True)

    return model


def train_causal_lm(
    train_tok,
    eval_tok,
    *,
    model_name: str = DEFAULT_DEEPSEEK_MODEL,
    output_dir: str = DEFAULT_DEEPSEEK_OUTPUT_DIR,
    max_steps: int = 80,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    experiment_name: str = "alpaca_llm_finetune",
    register_model_name: str | None = None,
    register_stage: str | None = None,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = True,
    cpu_threads: int | None = None,
) -> tuple[Any, str]:
    configure_cpu_threads(cpu_threads)

    try:
        import accelerate  # noqa: F401 — нужен для transformers.Trainer
    except ImportError as exc:
        raise ImportError(
            "Установите accelerate: %pip install 'accelerate>=1.1.0' "
            "и перезапустите kernel (Kernel → Restart)."
        ) from exc

    import mlflow
    import torch
    from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    mu = _reload_mlflow_utils()
    setup_mlflow = mu.setup_mlflow
    log_alpaca_causal_lm = mu.log_alpaca_causal_lm
    register_and_promote_run = mu.register_and_promote_run

    setup_mlflow(experiment_name=experiment_name)

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "PyTorch device:",
        "cuda" if torch.cuda.is_available() else "cpu",
        f"(cuda_available={torch.cuda.is_available()})",
    )
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "CPU threads:",
            f"torch={torch.get_num_threads()}",
            f"interop={torch.get_num_interop_threads()}",
        )

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print(f"tokenizer: loading {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"tokenizer: loaded {tokenizer.__class__.__name__}", flush=True)

    model = _load_causal_lm(
        model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        gradient_checkpointing=gradient_checkpointing,
        trust_remote_code=trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    logging_steps = max(1, min(10, max_steps))
    eval_steps = max(1, min(max_steps, max(1, max_steps // 4)))
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        logging_first_step=True,
        save_steps=max_steps,
        save_total_limit=1,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    class PrintLossCallback(TrainerCallback):
        """Печать train/eval loss в stdout на каждом logging step."""

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs:
                return
            parts: list[str] = []
            if "loss" in logs:
                parts.append(f"train_loss={float(logs['loss']):.4f}")
            if "eval_loss" in logs:
                parts.append(f"eval_loss={float(logs['eval_loss']):.4f}")
            if "learning_rate" in logs:
                parts.append(f"lr={float(logs['learning_rate']):.2e}")
            if parts:
                step = int(state.global_step)
                print(
                    f"[step {step}/{args.max_steps}] " + ", ".join(parts),
                    flush=True,
                )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        callbacks=[PrintLossCallback()],
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(
            {
                "model_name": model_name,
                "max_steps": max_steps,
                "output_dir": output_dir,
                "train_rows": len(train_tok),
                "eval_rows": len(eval_tok),
                "use_lora": use_lora,
                "lora_r": lora_r,
                "cpu_threads": cpu_threads,
            }
        )
        print(
            f"train: start (max_steps={max_steps}, logging every {logging_steps} step(s), "
            f"eval every {eval_steps} step(s))"
        )
        result = trainer.train()
        train_loss = float(result.training_loss)
        print(f"train: done — mean train_loss={train_loss:.4f}")
        print("evaluate: start")
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}={float(value):.4f}")
        mlflow.log_metrics(
            {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))}
        )
        print("evaluate: done")
        mlflow.log_metric("train_loss", train_loss)
        if eval_loss is not None:
            mlflow.log_metric("eval_loss", float(eval_loss))
        print("save: start")
        if use_lora:
            trainer.save_model(output_dir)
        else:
            model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print("save: done")
        print("mlflow log_model: start")
        log_alpaca_causal_lm(
            model=None if use_lora else model,
            tokenizer=None if use_lora else tokenizer,
            model_dir=output_dir,
            lightweight=use_lora,
        )
        print("mlflow log_model: done")
        run_id = run.info.run_id

    print(f"Локально: {output_dir}")
    print(f"MLflow run_id: {run_id}")

    if register_model_name:
        reg_name = register_model_name
    elif "deepseek" in model_name.lower():
        reg_name = DEEPSEEK_REGISTERED_MODEL_NAME
    elif "distilgpt" in model_name.lower():
        reg_name = DISTILGPT_REGISTERED_MODEL_NAME
    else:
        reg_name = ALPACA_REGISTERED_MODEL_NAME
    if register_stage:
        register_and_promote_run(
            run_id,
            model_name=reg_name,
            target_stage=register_stage,
        )
        serve_svc = (
            "mlflow-serve-deepseek"
            if reg_name == DEEPSEEK_REGISTERED_MODEL_NAME
            else "mlflow-serve-alpaca"
        )
        print(f"Serve: docker compose up -d {serve_svc} --force-recreate")

    return trainer, run_id


def train_both_causal_lm(
    train_sdf: DataFrame,
    eval_sdf: DataFrame,
    *,
    profiles: list[CausalLmTrainProfile] | None = None,
    max_steps: int = 500,
    register_stage: str | None = "Production",
    experiment_name: str = "alpaca_llm_finetune",
) -> dict[str, dict[str, Any]]:
    """Обучить distilgpt2 и DeepSeek на одном train/eval split (по очереди)."""
    from transformers import AutoTokenizer

    profs = profiles or default_train_profiles(max_steps=max_steps)
    results: dict[str, dict[str, Any]] = {}

    for profile in profs:
        print(f"\n{'=' * 60}")
        print(f"Обучение: {profile.label}  ({profile.model_name})")
        print(f"  → {profile.output_dir}")
        print(f"  → Registry: {profile.register_model_name}")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(
            profile.model_name, trust_remote_code=profile.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_tok, eval_tok = spark_texts_to_tokenized(
            train_sdf,
            eval_sdf,
            tokenizer,
            max_length=profile.max_length,
            model_name=profile.model_name,
        )
        print(f"tokenized: train={len(train_tok)}, eval={len(eval_tok)}")

        trainer, run_id = train_causal_lm(
            train_tok,
            eval_tok,
            model_name=profile.model_name,
            output_dir=profile.output_dir,
            max_steps=profile.max_steps,
            batch_size=profile.batch_size,
            learning_rate=profile.learning_rate,
            experiment_name=experiment_name,
            register_model_name=profile.register_model_name,
            register_stage=register_stage,
            use_lora=profile.use_lora,
            gradient_checkpointing=profile.gradient_checkpointing,
            trust_remote_code=profile.trust_remote_code,
            cpu_threads=profile.cpu_threads,
        )
        results[profile.label] = {
            "trainer": trainer,
            "run_id": run_id,
            "output_dir": profile.output_dir,
            "register_model_name": profile.register_model_name,
        }

    print(
        "\nГотово. Перезапуск serve:\n"
        "  docker compose up -d mlflow-serve-alpaca mlflow-serve-deepseek --force-recreate"
    )
    return results


def generate_sample(
    prompt: str,
    model_dir: str,
    *,
    max_new_tokens: int = 80,
    do_sample: bool = False,
    max_time: float | None = 120.0,
    allow_large_model: bool = False,
) -> str:
    """Генерация в формате Alpaca; обрезка по следующему ###."""
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_dir)
    adapter = path / "adapter_model.safetensors"
    checkpoint = path / "checkpoint-1"
    checkpoint_adapter = checkpoint / "adapter_model.safetensors"
    if (
        (not adapter.exists() or adapter.stat().st_size < 1024)
        and checkpoint_adapter.exists()
        and checkpoint_adapter.stat().st_size >= 1024
    ):
        path = checkpoint

    if (
        not allow_large_model
        and ("deepseek" in str(path).lower() or (path / "adapter_config.json").is_file())
    ):
        raise RuntimeError(
            "Локальная генерация DeepSeek/LoRA в Jupyter может уронить kernel. "
            "Используйте predict_deepseek_via_serve(..., serve_uri=uri) через порт 5003 "
            "или явно передайте allow_large_model=True."
        )

    cache_key = str(path.resolve())
    cache = globals().setdefault("_GENERATION_MODEL_CACHE", {})
    if cache_key in cache:
        tokenizer, model, device = cache[cache_key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(cache_key, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if (path / "adapter_config.json").is_file():
            from peft import PeftConfig, PeftModel

            peft_cfg = PeftConfig.from_pretrained(cache_key)
            base = AutoModelForCausalLM.from_pretrained(
                peft_cfg.base_model_name_or_path,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, cache_key)
        else:
            model = AutoModelForCausalLM.from_pretrained(cache_key, trust_remote_code=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        cache[cache_key] = (tokenizer, model, device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kw: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    }
    if max_time is not None:
        gen_kw["max_time"] = max_time
    if do_sample:
        gen_kw.update(do_sample=True, temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :].lstrip()
    for stop in ("\n\n###", "\n### Instruction", "\n### Input"):
        if stop in text:
            text = text.split(stop, 1)[0].rstrip()
    return text


def reload_alu():
    """Перезагрузить utils из файла (Jupyter). Использование: alu = reload_alu()"""
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    name = "alpaca_llm_utils"
    path = Path(__file__).resolve()
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

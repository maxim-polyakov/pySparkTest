#!/usr/bin/env python3
"""Скачать датасет с Hugging Face (snapshot всех файлов репозитория)."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        help="ID на HF, например tatsu-lab/alpaca",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Локальный каталог (по умолчанию: data/<имя-репо>)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.5,
        help="Минимум свободного места на диске (GB)",
    )
    args = parser.parse_args()

    repo_name = args.repo_id.split("/")[-1]
    default_dir = Path(__file__).resolve().parents[1] / "data" / repo_name
    out = (args.out or Path(os.environ.get("HF_DATASET_DIR", default_dir))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    need = int(args.min_free_gb * 1024**3)
    free = shutil.disk_usage(out).free
    if free < need:
        raise SystemExit(
            f"Недостаточно места в {out}: {free / 1e9:.2f} GB свободно, нужно ~{args.min_free_gb} GB.\n"
            "Укажите другой диск: --out B:\\datasets\\<name>"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("pip install -U huggingface_hub") from exc

    print(f"Downloading {args.repo_id} -> {out}")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out),
    )
    print(f"Done: {path}")


if __name__ == "__main__":
    main()

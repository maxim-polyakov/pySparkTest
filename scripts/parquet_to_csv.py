#!/usr/bin/env python3
"""Конвертация Parquet → CSV (один файл, каталог или glob)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _collect_parquet_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix != ".parquet":
            raise SystemExit(f"Не parquet-файл: {path}")
        return [path]
    if path.is_dir():
        pattern = "**/*.parquet" if recursive else "*.parquet"
        files = sorted(path.glob(pattern))
        if not files:
            raise SystemExit(f"Parquet не найден в {path}")
        return files
    raise SystemExit(f"Путь не существует: {path}")


def _csv_path(parquet: Path, out: Path | None, out_dir: Path | None) -> Path:
    if out is not None:
        if out.suffix.lower() == ".csv":
            return out
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{parquet.stem}.csv"
    base = out_dir or parquet.parent
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{parquet.stem}.csv"


def convert_one(parquet: Path, csv: Path, sep: str, encoding: str) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pip install pandas pyarrow") from exc

    print(f"  {parquet} -> {csv}")
    df = pd.read_parquet(parquet, engine="pyarrow")
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False, sep=sep, encoding=encoding)
    print(f"    строк: {len(df):,}, колонок: {len(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Файл .parquet, каталог с parquet или glob (например B:\\datasets\\alpaca\\data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Выходной .csv или каталог (для нескольких parquet)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Искать *.parquet в подкаталогах",
    )
    parser.add_argument("--sep", default=",", help="Разделитель CSV (по умолчанию запятая)")
    parser.add_argument("--encoding", default="utf-8", help="Кодировка CSV")
    args = parser.parse_args()

    input_path = args.input.resolve()
    files = _collect_parquet_files(input_path, args.recursive)

    out = args.out.resolve() if args.out else None
    if out and out.suffix.lower() == ".csv" and len(files) > 1:
        raise SystemExit("--out с расширением .csv можно указать только для одного parquet")

    out_dir = out if out and out.suffix.lower() != ".csv" else None

    print(f"Файлов parquet: {len(files)}")
    for parquet in files:
        csv = _csv_path(parquet, out, out_dir)
        convert_one(parquet, csv, args.sep, args.encoding)

    print("Готово.")


if __name__ == "__main__":
    main()

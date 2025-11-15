"""Herramienta rápida para diagnosticar las bases CSV de bbdd/.

Ejemplos:
    python inspect_csv.py bbdd/results_2021_comuna.csv
    python inspect_csv.py bbdd/results_2021_comuna.csv --head 3 --no-sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nowcast_snapshot import (
    RESULTS_2021_ALIASES,
    find_column,
    read_csv_guess,
    to_numeric,
)


def summarize_numeric_columns(df: pd.DataFrame) -> list[tuple[str, int, float]]:
    summary: list[tuple[str, int, float]] = []
    for col in df.columns:
        numeric = to_numeric(df[col])
        valid = int(numeric.notna().sum())
        if valid == 0:
            continue
        summary.append((col, valid, float(numeric.sum())))
    return summary


def explain_2021_aliases(df: pd.DataFrame) -> dict[str, str]:
    matches: dict[str, str] = {}
    for key, pats in RESULTS_2021_ALIASES.items():
        col = find_column(df, pats)
        if col:
            matches[key] = col
    return matches


def describe_file(path: Path, head: int, show_sample: bool) -> None:
    print("\n===", path, "===")
    df = read_csv_guess(path)
    print(f"Filas: {len(df):,}; Columnas: {len(df.columns)}")
    print("Columnas:")
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:>2}: {col}")
    summary = summarize_numeric_columns(df)
    if summary:
        print("Columnas con datos numéricos detectados:")
        total_rows = len(df)
        for col, count, total in summary:
            pct = count / total_rows * 100 if total_rows else 0
            print(f"  - {col}: {count:,} filas válidas ({pct:.1f}%), suma={total:,.2f}")
    else:
        print("No se detectaron columnas numéricas — revisa separadores o formateo.")
    if path.name == "results_2021_comuna.csv":
        matches = explain_2021_aliases(df)
        if matches:
            print("Mapeo resultados 2021 → columnas detectadas:")
            for key, col in matches.items():
                print(f"  · {key}: {col}")
        else:
            print("No se pudo mapear ninguna columna a los candidatos 2021.")
    if show_sample:
        print(f"\nPrimeras {min(head, len(df))} filas:")
        print(df.head(head))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico rápido para CSV del nowcast")
    parser.add_argument("paths", nargs="+", type=Path, help="Rutas de archivos a inspeccionar")
    parser.add_argument("--head", type=int, default=5, help="Número de filas a mostrar en la muestra")
    parser.add_argument("--no-sample", action="store_true", help="No mostrar la tabla de ejemplo")
    args = parser.parse_args()

    for path in args.paths:
        describe_file(path, head=args.head, show_sample=not args.no_sample)


if __name__ == "__main__":
    main()

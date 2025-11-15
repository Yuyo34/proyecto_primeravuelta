# ver_errores.py
# Uso:  python ver_errores.py [carpeta_csv]    # por defecto: ./bbdd

import sys, pandas as pd, numpy as np
from pathlib import Path

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bbdd")
print(f"📂 Leyendo desde: {BASE.resolve()}")

def leer(nombre):
    p = BASE / nombre
    for sep, dec in [( ",", "," ), ( ";", "," ), ( ",", "." ), ( ";", "." )]:
        try:
            df = pd.read_csv(p, encoding="utf-8", sep=sep, decimal=dec, low_memory=False)
            if df.shape[1] == 1 and sep == ",":  # separador incorrecto
                continue
            return df
        except Exception:
            continue
    print(f"❌ No se pudo leer {nombre}")
    return None

# ---------- 10) polls_long ----------
pl = leer("polls_long.csv")
if pl is None:
    sys.exit(1)

# a) ¿qué columnas hay y tipos básicos?
print("\n▶ polls_long: columnas y tipos")
print(pl.dtypes)

# b) encuestas cuya suma share_adj_pct ≠ 100 (tolerancia 1 pp)
if "share_adj_pct" in pl.columns and "poll_id" in pl.columns:
    tot = (pl.groupby("poll_id", dropna=False)["share_adj_pct"]
             .sum()
             .rename("suma_adj")
             .reset_index())
    bad = tot.loc[(tot["suma_adj"] - 100).abs() > 1]
    print("\n▶ ENCUESTAS con suma share_adj_pct ≠ 100 (>|1 pp|)")
    if bad.empty:
        print("  - Ninguna")
    else:
        print(bad.to_string(index=False))
        # Detalle de las primeras 2 encuestas problemáticas
        ej_ids = bad["poll_id"].dropna().astype(str).head(2).tolist()
        print("\n  Detalle (primeras 2):")
        for pid in ej_ids:
            print(f"\n  • poll_id = {pid}")
            cols = ["poll_id","house","candidate",
                    "share_reported_pct","indecisos_pct","share_adj_pct"]
            cols = [c for c in cols if c in pl.columns]
            print(pl.loc[pl["poll_id"].astype(str)==pid, cols].to_string(index=False))

# c) filas con poll_id vacío
if "poll_id" in pl.columns:
    vac = pl["poll_id"].isna().sum()
    if vac:
        print(f"\n▶ Filas con poll_id vacío: {vac}")
        print(pl.loc[pl["poll_id"].isna(), ["house","candidate","share_adj_pct"]].head(10).to_string(index=False))

# d) duplicados por clave (poll_id + candidate)
if {"poll_id","candidate"}.issubset(pl.columns):
    counts = (pl.groupby(["poll_id","candidate"]).size()
                .rename("n")
                .reset_index()
                .sort_values("n", ascending=False))
    dup_pairs = counts.loc[counts["n"] > 1]
    print("\n▶ DUPLICADOS por (poll_id, candidate)")
    print(f"  Total pares duplicados: {len(dup_pairs)}")
    print("  Top 10 con más repeticiones:")
    print(dup_pairs.head(10).to_string(index=False))

    # ejemplo: mostrar casas / fechas para el top-1 duplicado
    if len(dup_pairs):
        pid0, cand0 = dup_pairs.iloc[0][["poll_id","candidate"]]
        cols = ["poll_id","candidate","house","survey_date","n_effective","share_adj_pct","notes"]
        cols = [c for c in cols if c in pl.columns]
        print(f"\n  Detalle del top-1 duplicado: poll_id={pid0}, candidate={cand0}")
        print(pl.loc[(pl["poll_id"]==pid0) & (pl["candidate"]==cand0), cols]
                .sort_values(["house","survey_date"], ascending=[True, False])
                .to_string(index=False))

# e) w_poll: proporción en cero y resumen
if "w_poll" in pl.columns:
    w = pd.to_numeric(pl["w_poll"], errors="coerce")
    prop_cero = (w<=0).mean()
    print("\n▶ w_poll")
    print(f"  Filas: {len(w):,} ; % en cero/NA: {100*prop_cero:.2f}%")
    if (~w.isna()).any():
        print(w.describe(percentiles=[0.5,0.9,0.99]))

# f) valores no numéricos en columnas que deberían ser numéricas
num_cols = ["share_reported_pct","indecisos_pct","share_adj_pct","n_effective"]
print("\n▶ Valores no numéricos (coerción a NA) en columnas clave:")
for c in num_cols:
    if c in pl.columns:
        coer = pd.to_numeric(pl[c], errors="coerce")
        nn = coer.isna().sum()
        if nn and pl[c].notna().any():
            print(f"  - {c}: {nn} NA tras to_numeric (revisar strings/formatos)")

# ---------- 1) markets (si quieres ver coherencia bid/ask) ----------
mk = leer("markets.csv")
if mk is not None:
    print("\n▶ markets: columnas presentes")
    print(list(mk.columns))
    if {"best_bid","best_ask"}.issubset(mk.columns):
        bb = pd.to_numeric(mk["best_bid"], errors="coerce")
        ba = pd.to_numeric(mk["best_ask"], errors="coerce")
        bad = (bb.notna() & ba.notna() & (bb > ba))
        print(f"  Filas con bid>ask: {int(bad.sum())}")
        if bad.any():
            print(mk.loc[bad, ["market_id","candidate","best_bid","best_ask"]].head(10).to_string(index=False))

print("\nFin.")

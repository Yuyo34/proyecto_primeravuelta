# qc_csv.py
# Valida los CSV exportados desde Excel (ES-CL) y reporta problemas típicos.
# Uso:  python qc_csv.py [carpeta_csv]
# Por defecto lee desde .\bbdd

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ======== Config ========

# Carpeta por defecto donde están los CSV
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bbdd")

# Nombres de archivo SEGÚN tu carpeta (screenshot)
FILES = {
    "markets":                 "markets.csv",
    "market_weights":          "market_weights.csv",
    "results_2021_comuna":     "results_2021_comuna.csv",
    "pleb_2022_comuna":        "pleb_2022_comuna.csv",
    "pleb_2023_comuna":        "pleb_2023_comuna.csv",
    "comunas_estimacion_2024": "comunas_estimacion_2024.csv",
    "comunas_censo_2024":      "comunas_censo_2024.csv",
    "comuna_padron_2024":      "comuna_padron_2024.csv",
    "dinamics":                "dinamics.csv",          # (así aparece en tu carpeta)
    "polls_long":              "polls_long.csv",
    "house_bias_params":       "house_bias_params.csv",
    "house_effects":           "house_effects.csv",
}

# Candidatas a columnas de fecha por dataset (se convierten SOLO si existen)
DATE_CANDIDATES = {
    "markets":  ["ts_CLT", "timestamp", "ts", "fecha", "date", "datetime"],
    "polls_long": ["fieldwork_start","fieldwork_end","publish_date","survey_date","eval_time_CLT"],
    "dinamics": ["ts_CLT","fecha","date","datetime"],
}

# Muestra encabezados al cargar (útil para depurar nombres)
SHOW_COLUMNS = False

# ======== Utilidades ========

def read_csv_smart(path: Path) -> pd.DataFrame | None:
    """Lee CSV tolerando separador coma/; y coma/punto decimal.
       NO usa parse_dates para evitar errores si la columna no existe.
    """
    if not path.exists():
        print(f"❌ No existe: {path.name}")
        return None

    df = None
    # Intenta combinaciones comunes en ES-CL / Excel
    for sep, dec in [( ",", "," ), ( ";", "," ), ( ",", "." ), ( ";", "." )]:
        try:
            tmp = pd.read_csv(path, encoding="utf-8", sep=sep, decimal=dec)
            # Si leyó todo en 1 columna, probablemente el separador no era el correcto
            if tmp.shape[1] == 1 and sep == ",":
                continue
            df = tmp
            break
        except Exception:
            continue

    if df is None:
        print(f"❌ No se pudo leer: {path.name}")
        return None

    if SHOW_COLUMNS:
        print(f"   columnas {path.name}: {list(df.columns)}")
    return df

def to_dates_if_present(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce",
                                     dayfirst=True, infer_datetime_format=True)
    return df

def ok(msg):   print(f"✅ {msg}")
def warn(msg): print(f"⚠️  {msg}")
def info(msg): print(f"→ {msg}")

def standardize_cut(df: pd.DataFrame):
    if "cut_comuna" in df.columns:
        df["cut_comuna"] = df["cut_comuna"].astype(str).str.strip().str.zfill(5)
    return df

# ======== Carga ========

dfs: dict[str, pd.DataFrame] = {}
info(f"Leyendo CSV desde: {BASE.resolve()}")
for key, fname in FILES.items():
    p = BASE / fname
    df = read_csv_smart(p)
    if df is not None:
        # parsea fechas SOLO si hay columnas presentes
        if key in DATE_CANDIDATES:
            df = to_dates_if_present(df, DATE_CANDIDATES[key])
        dfs[key] = df
        ok(f"Cargado: {fname}  (filas={len(df):,})")
print()

# ======== Checks específicos ========

# 10) ENCUESTAS
pl = dfs.get("polls_long")
if pl is not None:
    info("[polls_long] Chequeos")
    req = {"poll_id","house","candidate","survey_date"}
    falt = req - set(pl.columns)
    if falt: warn(f"Faltan columnas mínimas: {falt}")

    # tipos numéricos típicos
    for c in ["share_reported_pct","indecisos_pct","share_adj_pct","w_poll","logit_adj"]:
        if c in pl: pl[c] = pd.to_numeric(pl[c], errors="coerce")

    # 100 por encuesta si existe share_adj_pct
    if {"poll_id","share_adj_pct"}.issubset(pl.columns):
        sums = pl.groupby("poll_id", dropna=False)["share_adj_pct"].sum().dropna()
        bad = (sums-100).abs() > 1.0
        ok("share_adj_pct cierra ≈100 por poll_id") if not bad.any() else warn(
            f"Suma share_adj_pct ≠100 (>|1 pp|) en {bad.sum()} encuestas. Ej: {list(sums[bad].index)[:5]}"
        )

    # duplicados poll_id+candidate
    if {"poll_id","candidate"}.issubset(pl.columns):
        dup = pl.duplicated(subset=["poll_id","candidate"], keep=False).sum()
        ok("Sin duplicados poll_id+candidate") if not dup else warn(
            f"Duplicados poll_id+candidate: {dup} filas"
        )

    if "survey_date" in pl:
        print(f"   rango survey_date: {pl['survey_date'].min()} → {pl['survey_date'].max()}")
    if "w_poll" in pl:
        s = pl["w_poll"].dropna()
        print(f"   ∑w_poll={s.sum():,.2f} ; filas w>0={(s>0).sum()}")
    if "logit_adj" in pl:
        s = pl["logit_adj"].dropna()
        if len(s):
            print(f"   rango logit_adj: [{s.min():.3f}, {s.max():.3f}]")
    print()

# 12) HOUSE EFFECTS
he = dfs.get("house_effects")
if he is not None:
    info("[house_effects] Chequeos")
    need = {"house","candidate","house_bias_logit"}
    falt = need - set(he.columns)
    if falt: warn(f"Faltan columnas: {falt}")
    he["house_bias_logit"] = pd.to_numeric(he.get("house_bias_logit"), errors="coerce")
    dup = he.duplicated(subset=["house","candidate"], keep=False).sum()
    ok("Claves únicas (house,candidate)") if not dup else warn(f"Claves duplicadas: {dup}")
    s = he["house_bias_logit"].dropna()
    if len(s):
        rng = (s.min(), s.max())
        print(f"   rango house_bias_logit: [{rng[0]:.3f}, {rng[1]:.3f}]")
        if s.abs().gt(1).any():
            warn("Sesgos >|1| logit; considera más shrink/HL para sesgos")
    print()

# 1) MARKETS
mk = dfs.get("markets")
if mk is not None:
    info("[markets] Chequeos")
    if {"best_bid","best_ask"}.issubset(mk.columns):
        bb = pd.to_numeric(mk["best_bid"], errors="coerce")
        ba = pd.to_numeric(mk["best_ask"], errors="coerce")
        bad = (bb.notna() & ba.notna()) & (bb > ba)
        ok("bid ≤ ask") if not bad.any() else warn(f"{bad.sum()} filas con best_bid > best_ask")
    # multi outcome presente?
    if "market_type" in mk.columns and "market_id" in mk.columns:
        n_multi = (mk["market_type"].astype(str).str.lower()=="multi").sum()
        if n_multi: ok(f"{n_multi} filas marcadas como multi-outcome (recuerda normalizar por market_id)")
    print()

# weights de mercado (simple presencia)
mw = dfs.get("market_weights")
if mw is not None:
    info("[market_weights] Chequeos básicos")
    print(f"   columnas: {list(mw.columns)}")
    print()

# COMUNAS (CUT consistente entre archivos)
for key in ["results_2021_comuna","pleb_2022_comuna","pleb_2023_comuna",
            "comunas_estimacion_2024","comunas_censo_2024","comuna_padron_2024"]:
    if key in dfs: dfs[key] = standardize_cut(dfs[key])

sets = [dfs.get(k) for k in ["results_2021_comuna","pleb_2022_comuna","pleb_2023_comuna",
                             "comunas_estimacion_2024","comunas_censo_2024","comuna_padron_2024"]
        if dfs.get(k) is not None]
if sets:
    base = set(sets[0]["cut_comuna"].dropna().unique()) if "cut_comuna" in sets[0].columns else set()
    for df in sets[1:]:
        if "cut_comuna" in df.columns:
            base = base & set(df["cut_comuna"].dropna().unique())
    if base:
        ok(f"Intersección consistente de CUT comunas: {len(base)}")

print("\nFin QC.")


from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("bbdd")
SRC  = BASE / "polls_long.csv"
DST  = BASE / "polls_long_clean.csv"

# --- Parámetros que puedes ajustar ---
DEFAULT_DEFF_ONLINE = 1.7  # usa el que estimes para panel online; o pon None para no imputar
CRITERIA_PID = "30102025_Criteria"

# Datos oficiales Criteria 30-oct-2025 (PDF)
CRITERIA_META = dict(
    house="Criteria",
    fieldwork_start="27-10-2025",
    fieldwork_end="29-10-2025",
    publish_date="30-10-2025",
    n_reported=1200,
    indecisos_pct=6.0,
    mode="Online",
    sample_type="No probabilístico",
)
CRITERIA_SHARE = {
    "Jeannette Jara":            27.0,
    "José Antonio Kast":         23.0,
    "Johannes Kaiser":           15.0,
    "Evelyn Matthei":            14.0,
    "Franco Parisi":              9.0,
    "Harold Mayne-Nicholls":      4.0,  # según tu evidencia
    "Marco Enríquez-Ominami":     2.0,  # nombre canónico
    "Eduardo Artés":              1.0,
}

# Canonización mínima de nombres de candidatos (agrega las que necesites)
CANONICAL_CANDIDATE = {
    "Marcos Enríquez-Ominami": "Marco Enríquez-Ominami",
    "Marco Enriquez-Ominami":  "Marco Enríquez-Ominami",
}

def detect_sep(p: Path) -> str:
    seps = [",",";","\t"]
    counts = {s:0 for s in seps}
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i>20: break
            for s in seps: counts[s] += line.count(s)
    return sorted(counts.items(), key=lambda kv: (-kv[1], ",;\t".index(kv[0])))[0][0]

def read_csv_robust(p: Path) -> pd.DataFrame:
    sep = detect_sep(p)
    try:
        return pd.read_csv(p, encoding="utf-8", sep=sep, decimal=",", low_memory=False)
    except pd.errors.ParserError as e:
        print(f"⚠️ ParserError con sep='{sep}': {e}")
        df = pd.read_csv(p, encoding="utf-8", sep=sep, decimal=",",
                         low_memory=False, engine="python", on_bad_lines="skip")
        print(f"⚠️ Se saltaron líneas problemáticas al leer {p.name}. Filas: {len(df):,}")
        return df

print(f"→ Leyendo {SRC.resolve()} ...")
pl = read_csv_robust(SRC)
pl.columns = [c.strip() for c in pl.columns]

# 1) Filas reales
mask_real = pl["poll_id"].notna() & pl["candidate"].notna()
pl = pl.loc[mask_real].copy()
pl.reset_index(drop=True, inplace=True)
print(f"Filas útiles tras limpiar poll_id/candidate: {len(pl):,}")

# 2) Canoniza nombres (evita duplicados por alias)
if "candidate" in pl.columns:
    pl["candidate"] = pl["candidate"].astype(str).str.strip().replace(CANONICAL_CANDIDATE)

# 3) Re-hidrata completamente Criteria: borra el poll y reinsertar 8 filas limpias
pl = pl[ pl["poll_id"] != CRITERIA_PID ].copy()

rows = []
for cand, pct in CRITERIA_SHARE.items():
    row = {
        "poll_id": CRITERIA_PID,
        **CRITERIA_META,
        "deff": np.nan,           # lo reconciliamos más abajo
        "n_effective": np.nan,    # idem
        "candidate": cand,
        "share_reported_pct": pct,
        "indecisos_pct": CRITERIA_META["indecisos_pct"],
        "notes": "Sobrescrito desde PDF oficial Criteria 30-oct-2025",
    }
    rows.append(row)
pl = pd.concat([pl, pd.DataFrame(rows)], ignore_index=True)

# 4) Recalcula share_adj_pct y renormaliza exacto a 100 por poll
for c in ["share_reported_pct","indecisos_pct"]:
    if c in pl.columns: pl[c] = pd.to_numeric(pl[c], errors="coerce")

def renorm(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    base = (100.0 - g["indecisos_pct"].astype(float))
    g["share_adj_pct"] = 100.0 * g["share_reported_pct"].astype(float) / base
    s = g["share_adj_pct"].sum()
    if pd.notna(s) and s != 0:
        g["share_adj_pct"] = g["share_adj_pct"] / s * 100.0
    return g

# Renormaliza por encuesta preservando poll_id (sin sorpresas de groupby.apply)
def renorm_by_poll(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    # dropna=False por si algún poll_id viniera vacío (no debería tras tu limpieza)
    for pid, g in df.groupby("poll_id", dropna=False):
        g2 = renorm(g)           # tu función renorm
        g2 = g2.copy()
        g2["poll_id"] = pid      # reinyecta el id del grupo
        parts.append(g2)
    return pd.concat(parts, ignore_index=True)

pl = renorm_by_poll(pl)

# 5) Reconciliar deff / n_effective de forma coherente
if "n_reported" in pl.columns: pl["n_reported"] = pd.to_numeric(pl["n_reported"], errors="coerce")
if "deff" in pl.columns:       pl["deff"]       = pd.to_numeric(pl["deff"], errors="coerce")
if "n_effective" in pl.columns: pl["n_effective"] = pd.to_numeric(pl["n_effective"], errors="coerce")

# a) si n_effective existe y n_reported>0 pero deff falta → deff = n_reported / n_effective
mask_calc_deff = pl["deff"].isna() & pl["n_effective"].notna() & (pl["n_reported"]>0)
pl.loc[mask_calc_deff, "deff"] = pl.loc[mask_calc_deff, "n_reported"] / pl.loc[mask_calc_deff, "n_effective"]

# b) si deff falta y NO hay n_effective pero sí n_reported → imputa deff por defecto (si definido)
if DEFAULT_DEFF_ONLINE is not None:
    mask_impute_deff = pl["deff"].isna() & pl["n_effective"].isna() & (pl["n_reported"]>0)
    pl.loc[mask_impute_deff, "deff"] = DEFAULT_DEFF_ONLINE

# c) con deff>0, recalcula n_effective
mask_neff = pl["deff"].notna() & (pl["deff"]>0) & (pl["n_reported"]>0)
pl.loc[mask_neff, "n_effective"] = pl.loc[mask_neff, "n_reported"] / pl.loc[mask_neff, "deff"]

# 6) Validación: todas las encuestas cierran en 100
chk = (pl.groupby("poll_id")["share_adj_pct"].sum() - 100).abs()
bad = chk[chk > 1e-6]
if len(bad):
    print("⚠️  Encuestas que no cierran a 100 tras renorm:")
    print(bad.head(10))
else:
    print("✅ Todas las encuestas cierran en 100 tras renormalizar.")

# 7) Vista Criteria final (solo para confirmar que NO quedó duplicado MEO)
cols_show = ["poll_id","house","fieldwork_start","fieldwork_end","publish_date",
             "n_reported","deff","n_effective","candidate","share_reported_pct",
             "indecisos_pct","share_adj_pct"]
print("\nVista Criteria (final):")
print(pl.loc[pl["poll_id"]==CRITERIA_PID, [c for c in cols_show if c in pl.columns]].to_string(index=False))

# 8) Guardar
pl.to_csv(DST, index=False, encoding="utf-8")
print(f"\n💾 Guardado: {DST.resolve()}")

# terrain_blend.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re, unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Parámetros ajustables
# =========================
BASE = Path("bbdd")

# archivo con cuotas nacionales (salida de nowcast_blend.py)
NATIONAL_FILES = [
    "nowcast_shares_poll_ESCL.csv",
    "nowcast_shares_poll.csv",
]

# pesos para el índice territorial L_c (0..1) usando plebiscitos
W_PLEB23_EN_CONTRA = 0.60
W_PLEB22_RECHAZO   = 0.40

# sensibilidad del tilt por candidato (más/menos sensibilidad a la “derecha/izquierda” territorial)
# valores >0: suben donde L_c alto (comunas más “en contra 23 / rechazo 22”), <0 lo contrario, 0 neutro
BETA_BY_CANDIDATE = {
    # ajusta estos nombres a los que vengan en tu archivo nacional
    "Jeannette Jara": -0.50,
    "José Antonio Kast": +0.50,
    "Johannes Kaiser": +0.40,
    "Evelyn Matthei": +0.20,
    "Franco Parisi":  0.00,
    "Harold Mayne-Nicholls": +0.10,
    "Marcos Enríquez-Ominami": -0.15,
    "Marco Enríquez-Ominami":  -0.15,  # alias posible
    "Eduardo Artés":  -0.30,
}

# Iteraciones y tolerancias del raking (IPF)
IPF_MAX_ITERS = 200
IPF_TOL = 1e-10

# =========================
# Utilidades robustas
# =========================
def _norm(s: str) -> str:
    """minúsculas, sin acentos, sólo [a-z0-9_] para comparar nombres/columnas"""
    s = str(s)
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def find_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """
    Devuelve la PRIMERA columna cuyo nombre normalizado matchee
    alguno de los patrones (regex sobre nombres normalizados).
    """
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}
    for pat in patterns:
        rx = re.compile(pat)
        for c in cols:
            if rx.search(norm[c]):
                return c
    return None

def read_csv_guess(path: Path) -> pd.DataFrame:
    """Lee CSV detectando separador (coma/; /|/tab) y BOM; falla si sigue en 1 sola columna."""
    for sep in [",", ";", "|", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    # último intento “duro”
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")
    if df.shape[1] == 1:
        raise RuntimeError(f"No pude detectar separador en {path}. ¿Es realmente CSV?")
    return df

def soft_map_name(name: str) -> str:
    """
    Normaliza/colapsa posibles variantes de nombres de candidato a una llave canónica.
    Útil para alinear archivos con rótulos levemente distintos.
    """
    n = _norm(name)
    aliases = {
        "marcos_enriquez_ominami": "Marcos Enríquez-Ominami",
        "marco_enriquez_ominami":  "Marcos Enríquez-Ominami",
        "jose_antonio_kast":      "José Antonio Kast",
        "jeannette_jara":         "Jeannette Jara",
        "evelyn_matthei":         "Evelyn Matthei",
        "franco_parisi":          "Franco Parisi",
        "johannes_kaiser":        "Johannes Kaiser",
        "harold_mayne_nicholls":  "Harold Mayne-Nicholls",
        "eduardo_artes":          "Eduardo Artés",
    }
    return aliases.get(n, name)

# =========================
# Lectura robusta del padrón y mapeo a comuna_id
# =========================
padron = read_csv_guess(BASE / "comuna_padron_2024.csv")
padron.rename(columns=lambda c: c.strip(), inplace=True)

reg_col  = find_col(padron, [r"^region$", "nombre_region"])
name_col = find_col(padron, [r"^comuna$", "nombre_comuna"])
pad_col  = find_col(padron, [r"padron", r"padr..n", r"habilitad", r"inscrit", r"electore"])

if name_col is None or reg_col is None:
    raise RuntimeError(
        "No encuentro columnas de 'region' y/o 'comuna' en comuna_padron_2024.csv. "
        f"Columnas vistas: {list(padron.columns)}"
    )
if pad_col is None:
    # tu archivo trae “pop_electores_2024”; úsalo si existe
    pad_col = find_col(padron, [r"pop_electore", r"electore"])
    if pad_col is None:
        raise RuntimeError(
            "No encuentro columna de padrón (padron/habilitados/inscritos/electores). "
            f"Columnas: {list(padron.columns)}"
        )

padron["region_nom"] = padron[reg_col].astype(str)
padron["comuna_nom"] = padron[name_col].astype(str)
padron["padron"]     = pd.to_numeric(padron[pad_col], errors="coerce").fillna(0)

# censo con comuna_id
censo = read_csv_guess(BASE / "comunas_censo_2024.csv")
censo.rename(columns=lambda c: c.strip(), inplace=True)

c_id   = find_col(censo, [r"^comuna_?id$", r"^cod", r"codigo_?comuna"])
c_reg  = find_col(censo, [r"^region", r"nombre_?region"])
c_name = find_col(censo, [r"^comuna$", r"nombre_?comuna"])

if c_id is None or c_name is None:
    raise RuntimeError(
        "No encuentro comuna_id y/o nombre de comuna en comunas_censo_2024.csv. "
        f"Columnas: {list(censo.columns)}"
    )

censo["region_nom"] = censo[c_reg].astype(str) if c_reg else ""
censo["comuna_nom"] = censo[c_name].astype(str)
censo["__key__"]    = (_norm(censo["region_nom"]) + "|" + _norm(censo["comuna_nom"]))

padron["__key__"]   = (_norm(padron["region_nom"]) + "|" + _norm(padron["comuna_nom"]))

mapa = censo[["__key__", c_id]].drop_duplicates()
mapa.columns = ["__key__", "comuna_id"]

padron = padron.merge(mapa, on="__key__", how="left")

faltan = padron["comuna_id"].isna().sum()
if faltan:
    # fallback: matchear sólo por comuna si región viene con abreviaturas distintas
    mapa2 = (censo.assign(__key2__=_norm(censo["comuna_nom"]))
                  [["__key2__", c_id]].drop_duplicates())
    mapa2.columns = ["__key2__", "comuna_id2"]
    padron["__key2__"] = _norm(padron["comuna_nom"])
    padron = padron.merge(mapa2, on="__key2__", how="left")
    padron["comuna_id"] = padron["comuna_id"].fillna(padron["comuna_id2"])
    padron.drop(columns=["comuna_id2","__key2__"], inplace=True)
    faltan = padron["comuna_id"].isna().sum()

if faltan:
    print(f"⚠️  Ojo: {faltan} comunas no encontraron ID. Revisa nombres/acentos en padrón.")

padron = padron[["comuna_id","region_nom","comuna_nom","padron"]].copy()
padron["comuna_id"] = padron["comuna_id"].astype("Int64")

# =========================
# Plebiscitos 2022 y 2023 → índice territorial L_c
# =========================
pleb22 = read_csv_guess(BASE / "pleb_2022_comuna.csv")
pleb23 = read_csv_guess(BASE / "pleb_2023_comuna.csv")

# detectar columnas (id + resultados)
p22_id = find_col(pleb22, [r"^comuna_?id$", r"^cod", r"codigo_?comuna"])
p23_id = find_col(pleb23, [r"^comuna_?id$", r"^cod", r"codigo_?comuna"])

rech22 = find_col(pleb22, [r"rechazo", r"en_?contra", r"contra"])
apr22  = find_col(pleb22, [r"apruebo", r"aprueba", r"^apr$"])

encontra23 = find_col(pleb23, [r"en_?contra", r"rechazo", r"contra"])
afavor23   = find_col(pleb23, [r"a_?favor", r"apruebo", r"aprueba"])

for nm, ref in [("p22_id",p22_id),("p23_id",p23_id),("rech22",rech22),("apr22",apr22),("encontra23",encontra23),("afavor23",afavor23)]:
    if ref is None:
        raise RuntimeError(f"No encuentro columna '{nm}' en plebiscitos. Revisa encabezados.")

# formatear
pleb22 = pleb22[[p22_id, rech22, apr22]].copy()
pleb22.columns = ["comuna_id","rechazo_2022","apruebo_2022"]

pleb23 = pleb23[[p23_id, encontra23, afavor23]].copy()
pleb23.columns = ["comuna_id","en_contra_2023","a_favor_2023"]

for c in ["rechazo_2022","apruebo_2022","en_contra_2023","a_favor_2023"]:
    pleb22[c] = pd.to_numeric(pleb22[c], errors="coerce") if c in pleb22.columns else pleb22
    pleb23[c] = pd.to_numeric(pleb23[c], errors="coerce") if c in pleb23.columns else pleb23

# merge base territorial
terr = padron.merge(pleb22, on="comuna_id", how="left").merge(pleb23, on="comuna_id", how="left")

# shares a 0..1 si vinieran en 0..100
def _to01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.max() is not None and s.max() > 1.5:
        s = s/100.0
    return s.clip(0,1)

terr["rechazo_2022"]   = _to01(terr["rechazo_2022"])
terr["en_contra_2023"] = _to01(terr["en_contra_2023"])

# índice territorial (0..1)
terr["L"] = (
    W_PLEB23_EN_CONTRA * terr["en_contra_2023"].fillna(terr["en_contra_2023"].mean())
  + W_PLEB22_RECHAZO   * terr["rechazo_2022"].fillna(terr["rechazo_2022"].mean())
)
# normalizamos 0..1 de nuevo por seguridad
mn, mx = terr["L"].min(), terr["L"].max()
if mx > mn:
    terr["L"] = (terr["L"] - mn) / (mx - mn)
else:
    terr["L"] = 0.5

# =========================
# Objetivo nacional (cuotas % por candidato)
# =========================
nat = None
for fn in NATIONAL_FILES:
    p = BASE / fn
    if p.exists():
        nat = pd.read_csv(p, encoding="utf-8-sig")
        break
if nat is None:
    raise RuntimeError(
        "No encontré archivo de cuotas nacionales. Genera primero nowcast_shares_poll_ESCL.csv "
        "con nowcast_blend.py (o deja nowcast_shares_poll.csv)."
    )

# columnas esperadas: candidate, share (0..100)
cand_col = find_col(nat, [r"^candidate$", r"candidato"])
share_col = find_col(nat, [r"^share$", r"cuota", r"porcentaje"])

if cand_col is None or share_col is None:
    raise RuntimeError(f"Archivo nacional {fn} no tiene columnas candidate/share. Columnas: {list(nat.columns)}")

nat = nat[[cand_col, share_col]].copy()
nat.columns = ["candidate", "share"]
nat["candidate"] = nat["candidate"].map(soft_map_name)
nat["share"] = pd.to_numeric(nat["share"], errors="coerce").fillna(0.0)

# normalizar a 1.0
S = nat.copy()
if S["share"].max() > 1.5:
    S["share"] = S["share"] / 100.0
S = S[S["share"]>0].reset_index(drop=True)

candidates = list(S["candidate"])
s_nat = S.set_index("candidate")["share"].to_dict()

# =========================
# Tilt territorial + raking para cuadrar nacional
# =========================
# base: todos con la misma cuota nacional
base = terr[["comuna_id","padron","L"]].copy()

# tilt multiplicativo por candidato
L = base["L"].values
L_center = L.mean()
shares = {}

for cand in candidates:
    beta = BETA_BY_CANDIDATE.get(cand, 0.0)
    # multiplicador exponencial del tilt
    m = np.exp(beta * (L - L_center))
    # aplicar multiplicador al share nacional y renormalizar por comuna
    shares[cand] = m * s_nat[cand]

df_sh = pd.DataFrame(shares, index=base.index)  # comunas × candidatos
# renormalizar para que por comuna sumen 1
row_sum = df_sh.sum(axis=1).replace(0, np.nan)
df_sh = df_sh.div(row_sum, axis=0).fillna(1.0/len(candidates))

# raking (IPF) para que la suma ponderada por padrón cuadre con s_nat
w = base["padron"].astype(float).values
target = np.array([s_nat[c] for c in candidates])

def ipf(matrix: np.ndarray, weights: np.ndarray, target: np.ndarray,
        max_iters=IPF_MAX_ITERS, tol=IPF_TOL) -> np.ndarray:
    X = matrix.copy()
    for _ in range(max_iters):
        # columnas (candidatos) → ajustar para que cumplan objetivo nacional
        current = (weights[:, None] * X).sum(axis=0)
        scale = np.divide(target, current, out=np.ones_like(target), where=current>0)
        X *= scale

        # renormalizar filas (comunas) para que cada fila sea una distribución
        X = np.divide(X, X.sum(axis=1, keepdims=True), out=X, where=X.sum(axis=1, keepdims=True)>0)

        err = np.abs((weights[:, None] * X).sum(axis=0) - target).max()
        if err < tol:
            break
    return X

X0 = df_sh.values
X = ipf(X0, w, target)
df_final = pd.DataFrame(X, columns=candidates, index=base.index)

# a % y con ids
df_final = df_final.mul(100)
out_communal = pd.concat([base[["comuna_id","padron"]].reset_index(drop=True), df_final.reset_index(drop=True)], axis=1)

# guardamos comunas × candidatos
out_communal.to_csv(BASE / "comunas_nowcast_shares.csv", index=False, encoding="utf-8-sig")

# agregamos nacional para verificación
agg_nat = (out_communal[candidates].multiply(out_communal["padron"], axis=0).sum()
           / out_communal["padron"].sum())
out_nat = pd.DataFrame({
    "candidate": candidates,
    "share_from_terrain": agg_nat.values,  # ya está en %
})
out_nat.to_csv(BASE / "nowcast_shares_nacional_from_terrain.csv", index=False, encoding="utf-8-sig")

print("\nListo:")
print(" - bbdd/comunas_nowcast_shares.csv")
print(" - bbdd/nowcast_shares_nacional_from_terrain.csv")

print("\nSanity checks:")
print(f"  • Suma shares nacionales (entrada) = {S['share'].sum()*100:.1f}%")
print(f"  • Suma shares nacionales (salida)  = {out_nat['share_from_terrain'].sum():.1f}%")
mx, mn = out_communal[candidates].max().max(), out_communal[candidates].min().min()
print(f"  • Rangos comunales (0–100) = {mn:.3f} → {mx:.3f}")

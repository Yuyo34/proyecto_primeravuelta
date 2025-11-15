# nowcast_blend.py
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("bbdd")

# ------------------- PARÁMETROS AJUSTABLES -------------------
LAMBDA = 0.60              # peso del blend: 60% encuestas, 40% mercados
K_CAP = 4000               # tope de concentración para la Dirichlet (evita sobreconfianza)
N_SIMS = 200_000           # nº simulaciones Monte Carlo para p_win_from_polls
POLL_WEIGHT_COLS = ["w_logit", "w_poll", "w_pre", "w_time", "w_size"]  # prioridad de pesos
RANDOM_SEED = 42           # semilla para reproducibilidad
# Sensibilidad (puedes ajustar)
LAMBDA_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
K_GRID = [1000, 2000, 3000, 4000, 6000]
# -------------------------------------------------------------

def prefer_clean(name: str) -> Path:
    """Usa polls_long_clean.csv si existe; si no, polls_long.csv."""
    if name == "polls_long" and (BASE / "polls_long_clean.csv").exists():
        return BASE / "polls_long_clean.csv"
    return BASE / f"{name}.csv"

# -------- lectura robusta (detecta ; , \t y coma decimal) ----------
def detect_sep(path: Path, sample_lines: int = 30) -> str:
    seps = [",", ";", "\t"]
    counts = {s: 0 for s in seps}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            for s in seps:
                counts[s] += line.count(s)
    order = {",": 0, ";": 1, "\t": 2}  # desempate
    return sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0][0]

def read_csv_smart(name: str) -> pd.DataFrame:
    """Lee CSV detectando separador y coma decimal, con fallback estable."""
    p = prefer_clean(name)
    sep = detect_sep(p)
    for dec in [",", "."]:
        try:
            df = pd.read_csv(p, sep=sep, decimal=dec, encoding="utf-8", low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:
            continue
    df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df
# -------------------------------------------------------------------

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def inv_logit(z):
    return 1 / (1 + np.exp(-z))

# ------------------- LECTURAS -------------------
polls   = read_csv_smart("polls_long")
houses  = read_csv_smart("house_effects")
markets = read_csv_smart("markets")
mktw    = read_csv_smart("market_weights") if (BASE / "market_weights.csv").exists() else pd.DataFrame(columns=["platform","BSS_norm"])

# ------------------- LIMPIEZAS ENCUESTAS -------------------
polls = polls[polls["poll_id"].notna() & polls["candidate"].notna()].copy()
polls.columns = [c.strip() for c in polls.columns]
for c in ["share_adj_pct", "n_effective", "n_reported"]:
    if c in polls.columns:
        polls[c] = pd.to_numeric(polls[c], errors="coerce")

# Peso de encuesta (elige el primero disponible y útil)
if not any(col in polls.columns for col in POLL_WEIGHT_COLS):
    polls["w_use"] = 1.0
else:
    for col in POLL_WEIGHT_COLS:
        if col in polls.columns:
            w = pd.to_numeric(polls[col], errors="coerce")
            if (np.nan_to_num(w, nan=0.0) > 0).any():
                polls["w_use"] = w.fillna(0.0)
                break
    if "w_use" not in polls.columns:
        polls["w_use"] = 1.0

# ------------------- SESGOS POR CASA (robusto) -------------------
houses_cols_orig = list(houses.columns)
houses = houses.rename(columns={c: c.strip() for c in houses.columns})
rename_map = {
    "candidato": "candidate",
    "candidata": "candidate",
    "encuestadora": "house",
    "casa": "house",
    "pollster": "house",
    "sesgo_logit": "house_bias_logit",
    "bias_logit": "house_bias_logit",
    "house_bias": "house_bias_logit",
    "bias": "house_bias_logit",
}
houses = houses.rename(columns=rename_map)
for needed in ["house", "candidate"]:
    if needed not in houses.columns:
        houses = pd.DataFrame(columns=["house","candidate","house_bias_logit"])
        break
if not houses.empty:
    cols_keep = [c for c in ["house","candidate","house_bias_logit"] if c in houses.columns]
    houses = houses[cols_keep].copy()
    if "house_bias_logit" in houses.columns:
        houses["house_bias_logit"] = pd.to_numeric(houses["house_bias_logit"], errors="coerce")
else:
    houses = pd.DataFrame(columns=["house","candidate","house_bias_logit"])
polls = polls.merge(houses, on=["house","candidate"], how="left")
if "house_bias_logit" not in polls.columns:
    polls["house_bias_logit"] = 0.0
else:
    polls["house_bias_logit"] = pd.to_numeric(polls["house_bias_logit"], errors="coerce").fillna(0.0)

# ------------------- ENCUESTAS → SHARE -------------------
polls["p_share"]   = polls["share_adj_pct"] / 100.0
polls["logit_raw"] = logit(polls["p_share"])
polls["logit_adj"] = polls["logit_raw"] - polls["house_bias_logit"]
grp = polls.groupby("candidate", as_index=False).agg(
    sum_w=("w_use","sum"),
    sum_wl=("logit_adj", lambda s: np.sum(s * polls.loc[s.index, "w_use"]))
)
grp["logit_poll"]     = grp["sum_wl"] / grp["sum_w"]
grp["share_poll"]     = inv_logit(grp["logit_poll"])
grp["share_poll_pct"] = grp["share_poll"] * 100.0
total = grp["share_poll_pct"].sum()
if total > 0:
    grp["share_poll_pct"] = grp["share_poll_pct"] / total * 100.0
shares_out = grp[["candidate","share_poll_pct","sum_w"]].sort_values("share_poll_pct", ascending=False)

# ------------------- MERCADOS → P_WIN -------------------
def parse_prob_cell(x):
    """Admite '92%', '0,92', '0.92', 92, 0.92 → [0,1]."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "").replace(",", ".")
    try: v = float(s)
    except Exception: return np.nan
    return v/100.0 if v > 1.0 else v

markets.columns = [c.strip() for c in markets.columns]
if "implied_prob" in markets.columns:
    markets["implied_prob"] = markets["implied_prob"].apply(parse_prob_cell)
elif "price_yes" in markets.columns:
    markets["implied_prob"] = (
        markets["price_yes"].astype(str).str.replace(",", ".", regex=False).astype(float) / 100.0
    )
else:
    raise SystemExit("markets.csv no tiene 'implied_prob' ni 'price_yes'.")
if "platform" not in markets.columns:
    markets["platform"] = "unknown"
if "datetime_CLT" in markets.columns:
    markets["datetime_CLT"] = pd.to_datetime(markets["datetime_CLT"], errors="coerce", dayfirst=True)

# Pesos por plataforma
if not mktw.empty and "BSS_norm" in mktw.columns and "platform" in mktw.columns:
    wtab = mktw[["platform","BSS_norm"]].dropna().copy()
    wtab["w_platform"] = pd.to_numeric(wtab["BSS_norm"], errors="coerce").clip(lower=1e-6)
    wtab["w_platform"] = wtab["w_platform"] / (wtab["w_platform"].sum() if wtab["w_platform"].sum() > 0 else 1.0)
    markets = markets.merge(wtab[["platform","w_platform"]], on="platform", how="left").fillna({"w_platform":1.0})
else:
    markets["w_platform"] = 1.0

# Última observación por (candidate, platform) y promedio ponderado
if "datetime_CLT" in markets.columns:
    markets = markets.sort_values("datetime_CLT")
    last = markets.groupby(["candidate","platform"], as_index=False).tail(1)
else:
    last = markets.copy()
tmp = last.copy()
tmp = tmp.loc[tmp["implied_prob"].notna()].copy()
tmp["num"] = pd.to_numeric(tmp["implied_prob"], errors="coerce") * pd.to_numeric(tmp["w_platform"], errors="coerce")
agg = tmp.groupby("candidate", as_index=False).agg(
    sum_num=("num", "sum"),
    sum_den=("w_platform", "sum")
)
agg["p_win_market"] = (agg["sum_num"] / agg["sum_den"]).clip(lower=0, upper=1).fillna(0.0)
mkt_out = agg[["candidate"]].copy()
mkt_out["p_win_market_pct"] = agg["p_win_market"] * 100.0
mkt_out = mkt_out.sort_values("p_win_market_pct", ascending=False)

# ------------------- ENCUESTAS → P_WIN (Monte Carlo) -------------------
if "n_effective" in polls.columns:
    neff_per_poll = polls.drop_duplicates("poll_id")[["poll_id","n_effective"]].copy()
    K_eff = pd.to_numeric(neff_per_poll["n_effective"], errors="coerce").fillna(0).sum()
else:
    K_eff = 1000
K = int(min(max(K_eff, 200), K_CAP))
p_vec = shares_out["share_poll_pct"].values / 100.0
cands = shares_out["candidate"].tolist()

alpha = p_vec * K
rng = np.random.default_rng(RANDOM_SEED)
draws = rng.dirichlet(alpha, size=N_SIMS)  # (N_SIMS, n_candidatos)
winners = np.argmax(draws, axis=1)
counts = np.bincount(winners, minlength=len(cands))
p_win_polls = counts / counts.sum()
polls_win_out = pd.DataFrame({"candidate": cands, "p_win_from_polls_pct": p_win_polls * 100.0})
polls_win_out = polls_win_out.sort_values("p_win_from_polls_pct", ascending=False)

# ------------------- BLEND FINAL -------------------
blend = polls_win_out.merge(mkt_out, on="candidate", how="outer").fillna(0.0)
blend["p_win_blend_pct"] = LAMBDA * blend["p_win_from_polls_pct"] + (1 - LAMBDA) * blend["p_win_market_pct"]
blend = blend.sort_values("p_win_blend_pct", ascending=False)

# ------------------- GUARDAR (CSV estándar) -------------------
(shares_out[["candidate","share_poll_pct","sum_w"]]).to_csv(BASE / "nowcast_shares_poll.csv", index=False, encoding="utf-8")
mkt_out.to_csv(BASE / "winner_probs_market.csv", index=False, encoding="utf-8")
polls_win_out.to_csv(BASE / "winner_probs_polls.csv", index=False, encoding="utf-8")
blend.to_csv(BASE / "winner_probs_blend.csv", index=False, encoding="utf-8")

# ---- Export ES-CL (punto y coma + coma decimal) ----
def save_csv_chile(df: pd.DataFrame, path: Path, float_cols: list[str] | None = None):
    out = df.copy()
    if float_cols is None:
        float_cols = out.select_dtypes(include=["number"]).columns.tolist()
    for c in float_cols:
        out[c] = out[c].map(lambda x: (f"{x:.6f}".replace(".", ",") if pd.notna(x) else ""))
    out.to_csv(path, sep=";", index=False, encoding="utf-8-sig")

save_csv_chile(shares_out[["candidate","share_poll_pct","sum_w"]], BASE / "nowcast_shares_poll_ESCL.csv", ["share_poll_pct","sum_w"])
save_csv_chile(mkt_out,            BASE / "winner_probs_market_ESCL.csv", ["p_win_market_pct"])
save_csv_chile(polls_win_out,      BASE / "winner_probs_polls_ESCL.csv",  ["p_win_from_polls_pct"])
save_csv_chile(blend,              BASE / "winner_probs_blend_ESCL.csv",  ["p_win_from_polls_pct","p_win_market_pct","p_win_blend_pct"])

# ===================== A) TABLA ÚNICA FINAL =========================
final_table = shares_out.rename(columns={"share_poll_pct":"share_from_polls_pct"})[["candidate","share_from_polls_pct"]]
final_table = final_table.merge(mkt_out, on="candidate", how="outer")
final_table = final_table.merge(polls_win_out, on="candidate", how="outer")
final_table = final_table.merge(blend[["candidate","p_win_blend_pct"]], on="candidate", how="outer")
final_table = final_table.fillna(0.0).sort_values("p_win_blend_pct", ascending=False)
final_table.to_csv(BASE / "nowcast_final_table.csv", index=False, encoding="utf-8")
save_csv_chile(final_table, BASE / "nowcast_final_table_ESCL.csv",
               ["share_from_polls_pct","p_win_market_pct","p_win_from_polls_pct","p_win_blend_pct"])

# ===================== B) SENSIBILIDAD (LAMBDA, K) ==================
def blend_with_params(lambda_val: float, K_val: int):
    alpha_test = p_vec * K_val
    rng2 = np.random.default_rng(RANDOM_SEED)
    draws2 = rng2.dirichlet(alpha_test, size=N_SIMS)
    winners2 = np.argmax(draws2, axis=1)
    counts2 = np.bincount(winners2, minlength=len(cands))
    p_win_polls2 = counts2 / counts2.sum()
    polls_win2 = pd.DataFrame({"candidate": cands, "p_win_from_polls_pct": p_win_polls2 * 100.0})
    blend2 = polls_win2.merge(mkt_out, on="candidate", how="outer").fillna(0.0)
    blend2["p_win_blend_pct"] = lambda_val * blend2["p_win_from_polls_pct"] + (1 - lambda_val) * blend2["p_win_market_pct"]
    top_row = blend2.sort_values("p_win_blend_pct", ascending=False).iloc[0]
    return top_row["candidate"], float(top_row["p_win_blend_pct"])

rows = []
for lam in LAMBDA_GRID:
    for Ktest in K_GRID:
        Ktest_use = int(min(Ktest, max(K_CAP, 4000)))  # techo suave
        leader, p_blend = blend_with_params(lam, Ktest_use)
        rows.append({"LAMBDA": lam, "K": Ktest_use, "leader": leader, "p_win_blend_pct": p_blend})
sens = pd.DataFrame(rows).sort_values(["LAMBDA","K"])
sens.to_csv(BASE / "nowcast_sensitivity.csv", index=False, encoding="utf-8")
save_csv_chile(sens, BASE / "nowcast_sensitivity_ESCL.csv", ["LAMBDA","K","p_win_blend_pct"])

# ===================== C) PROBABILIDAD TOP-2 ========================
top2_idx = np.argpartition(draws, -2, axis=1)[:, -2:]
top2_counts = np.zeros(len(cands), dtype=np.int64)
for i in range(len(cands)):
    top2_counts[i] = np.sum((top2_idx == i).any(axis=1))
p_top2 = top2_counts / N_SIMS
top2_out = pd.DataFrame({"candidate": cands, "p_top2_pct": p_top2 * 100.0}).sort_values("p_top2_pct", ascending=False)
top2_out.to_csv(BASE / "nowcast_top2_probs.csv", index=False, encoding="utf-8")
save_csv_chile(top2_out, BASE / "nowcast_top2_probs_ESCL.csv", ["p_top2_pct"])

# ---- Sanity checks y resumen de archivos ----
print("\nListo:")
for f in [
    "nowcast_shares_poll.csv",
    "winner_probs_market.csv",
    "winner_probs_polls.csv",
    "winner_probs_blend.csv",
    "nowcast_shares_poll_ESCL.csv",
    "winner_probs_market_ESCL.csv",
    "winner_probs_polls_ESCL.csv",
    "winner_probs_blend_ESCL.csv",
    "nowcast_final_table.csv",
    "nowcast_final_table_ESCL.csv",
    "nowcast_sensitivity.csv",
    "nowcast_sensitivity_ESCL.csv",
    "nowcast_top2_probs.csv",
    "nowcast_top2_probs_ESCL.csv",
]:
    print(" -", str(BASE / f))

print(f"\nParámetros base: LAMBDA={LAMBDA:.2f}, K={K} (cap {K_CAP}), N_SIMS={N_SIMS:,}, seed={RANDOM_SEED}")
print("\nSanity checks:")
print("  • Suma shares encuestas = ", round(shares_out['share_poll_pct'].sum(), 4))
rngs = blend[["p_win_from_polls_pct","p_win_market_pct","p_win_blend_pct"]]
print("  • Rangos blend 0-100: ", float(rngs.min().min()), "→", float(rngs.max().max()))

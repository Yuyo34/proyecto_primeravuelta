"""Construye el nowcast nacional usando las bases reales del repositorio.

El flujo replica la descripción de ``modelo_nowcast_presidencial.md``:
1. Prior histórico con 2021 + plebiscitos 2022/2023 → Dirichlet.
2. Pseudo-votos de encuestas (``polls_long_clean.csv`` si existe, si no ``polls_long.csv``).
3. Probabilidades implícitas de mercados (``markets.csv`` + ``market_weights.csv`` opcional).
4. Simulación Monte-Carlo para ``p_win_from_polls`` y mezcla con mercados (``λ``).
5. Exporta ``bbdd/nowcast_final_table.csv`` y el gráfico ``salidas/grafico_intencion_voto.png``.
"""
from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("bbdd")
OUT = Path("salidas")

POLL_WEIGHT_TARGET = 4_500   # fuerza efectiva asignada a las encuestas
MARKET_WEIGHT_TARGET = 1_800 # fuerza equivalente para mercados
PRIOR_WEIGHT_TARGET = 2_400  # peso total del prior histórico (2021 + plebis)
LAMBDA = 0.60                # mezcla prob. encuestas vs mercados
N_SIMS = 200_000
RNG_SEED = 2025

# --- utilidades generales -------------------------------------------------

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def detect_sep(path: Path, sample: int = 30) -> str:
    seps = [",", ";", "|", "\t"]
    counts = {s: 0 for s in seps}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i >= sample:
                break
            for s in seps:
                counts[s] += line.count(s)
    return max(seps, key=lambda s: (counts[s], -seps.index(s)))


def read_csv_guess(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path} — sube la base a bbdd/ según el README.")
    sep = detect_sep(path)
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", engine="python")
    except Exception as exc:
        raise RuntimeError(f"No pude leer {path.name} (sep='{sep}'): {exc}")
    df.columns = [c.strip() for c in df.columns]
    return df


NUMERIC_JUNK_RE = re.compile(r"[^0-9,\.\-]+")


def _normalize_number_string(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    value = NUMERIC_JUNK_RE.sub("", value)
    if not value:
        return ""
    comma_pos = value.rfind(",")
    dot_pos = value.rfind(".")
    if "," in value and "." in value:
        if comma_pos > dot_pos:
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    else:
        if value.count(",") == 1 and "." not in value:
            value = value.replace(",", ".")
        elif value.count(".") > 1 and "," not in value:
            value = value.replace(".", "")
    return value


def to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    normalized = cleaned.map(_normalize_number_string)
    return pd.to_numeric(normalized, errors="coerce")


def prefer_clean(name: str) -> Path:
    if name == "polls_long" and (BASE / "polls_long_clean.csv").exists():
        return BASE / "polls_long_clean.csv"
    return BASE / f"{name}.csv"


# --- prior histórico: 2021 + plebis --------------------------------------

RESULTS_2021_ALIASES: dict[str, list[str]] = {
    "gabriel_boric":       [r"boric"],
    "jose_antonio_kast":   [r"kast"],
    "sebastian_sichel":    [r"sichel"],
    "yasna_provoste":      [r"provoste"],
    "franco_parisi":       [r"parisi"],
    "marco_enriquez":      [r"marco", r"enriquez", r"ominami"],
    "eduardo_artes":       [r"artes"],
    "otros":               [r"otros", r"otro"],
}


def find_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for pat in patterns:
        rx = re.compile(pat)
        for norm_name, original in norm_map.items():
            if rx.search(norm_name):
                return original
    return None


def aggregate_results_2021() -> dict[str, float]:
    df = read_csv_guess(BASE / "results_2021_comuna.csv")
    shares: dict[str, float] = {}
    matched_columns: dict[str, str] = {}
    for key, pats in RESULTS_2021_ALIASES.items():
        col = find_column(df, pats)
        if not col:
            continue
        matched_columns[key] = col
        shares[key] = float(to_numeric(df[col]).fillna(0.0).sum())
    total = sum(shares.values())
    if total <= 0:
        cols = ", ".join(df.columns[:20])  # evita imprimir archivos enormes
        raise RuntimeError(
            "results_2021_comuna.csv no tiene columnas numéricas reconocibles. "
            f"Columnas detectadas: {matched_columns or 'ninguna'}. Encabezados disponibles: {cols}"
        )
        shares[key] = float(to_numeric(df[col]).fillna(0.0).sum())
    total = sum(shares.values())
    if total <= 0:
        raise RuntimeError("results_2021_comuna.csv no tiene columnas numéricas reconocibles.")
    for key in list(shares):
        shares[key] /= total
    block = {
        "right_2021": shares.get("jose_antonio_kast", 0.0) + shares.get("sebastian_sichel", 0.0),
        "left_2021": shares.get("gabriel_boric", 0.0) + shares.get("yasna_provoste", 0.0)
        + shares.get("marco_enriquez", 0.0) + shares.get("eduardo_artes", 0.0),
        "populist_2021": shares.get("franco_parisi", 0.0),
    }
    rem = max(0.0, 1.0 - sum(block.values()))
    block["others_2021"] = rem
    return block


def aggregate_plebiscite(path: Path, positive: list[str], negative: list[str]) -> tuple[float, float]:
    df = read_csv_guess(path)
    pos_col = find_column(df, positive)
    neg_col = find_column(df, negative)
    if not pos_col or not neg_col:
        raise RuntimeError(f"No pude detectar columnas plebiscito en {path.name}.")
    pos = to_numeric(df[pos_col]).fillna(0.0).sum()
    neg = to_numeric(df[neg_col]).fillna(0.0).sum()
    total = pos + neg
    if total <= 0:
        raise RuntimeError(f"{path.name} no tiene datos numéricos válidos.")
    return pos / total, neg / total


def build_prior_blocks() -> dict[str, float]:
    blocks = aggregate_results_2021()
    rech22, apr22 = aggregate_plebiscite(
        BASE / "pleb_2022_comuna.csv",
        positive=[r"rechazo", r"en_contra"],
        negative=[r"apruebo", r"a_favor"],
    )
    encon23, afav23 = aggregate_plebiscite(
        BASE / "pleb_2023_comuna.csv",
        positive=[r"en_contra", r"rechazo"],
        negative=[r"a_favor", r"apruebo"],
    )
    blocks.update(
        {
            "pleb_rechazo_2022": rech22,
            "pleb_apruebo_2022": apr22,
            "pleb_en_contra_2023": encon23,
            "pleb_a_favor_2023": afav23,
        }
    )
    return blocks


PRIOR_WEIGHTS_BY_CANDIDATE: dict[str, dict[str, float]] = {
    "Jeannette Jara": {
        "left_2021": 0.45,
        "pleb_apruebo_2022": 0.30,
        "pleb_a_favor_2023": 0.25,
    },
    "José Antonio Kast": {
        "right_2021": 0.55,
        "pleb_rechazo_2022": 0.25,
        "pleb_en_contra_2023": 0.20,
    },
    "Johannes Kaiser": {
        "right_2021": 0.60,
        "pleb_rechazo_2022": 0.20,
        "pleb_en_contra_2023": 0.20,
    },
    "Evelyn Matthei": {
        "right_2021": 0.35,
        "populist_2021": 0.25,
        "pleb_rechazo_2022": 0.20,
        "pleb_en_contra_2023": 0.20,
    },
    "Franco Parisi": {
        "populist_2021": 0.70,
        "right_2021": 0.15,
        "left_2021": 0.15,
    },
    "Harold Mayne-Nicholls": {
        "left_2021": 0.30,
        "populist_2021": 0.30,
        "pleb_apruebo_2022": 0.20,
        "pleb_a_favor_2023": 0.20,
    },
    "Marco Enríquez-Ominami": {
        "left_2021": 0.55,
        "pleb_apruebo_2022": 0.25,
        "pleb_a_favor_2023": 0.20,
    },
    "Eduardo Artés": {
        "left_2021": 0.60,
        "pleb_apruebo_2022": 0.20,
        "pleb_a_favor_2023": 0.20,
    },
}

DEFAULT_PRIOR_WEIGHTS = {
    "left_2021": 0.30,
    "right_2021": 0.30,
    "populist_2021": 0.15,
    "pleb_apruebo_2022": 0.125,
    "pleb_rechazo_2022": 0.125,
    "pleb_a_favor_2023": 0.035,
    "pleb_en_contra_2023": 0.035,
}


def build_prior_alpha(candidates: list[str], blocks: dict[str, float]) -> pd.Series:
    raw = {}
    for cand in candidates:
        weights = PRIOR_WEIGHTS_BY_CANDIDATE.get(cand, DEFAULT_PRIOR_WEIGHTS)
        denom = sum(weights.values()) or 1.0
        val = 0.0
        for block_name, w in weights.items():
            val += (w / denom) * blocks.get(block_name, 0.0)
        raw[cand] = val
    total = sum(raw.values()) or 1.0
    shares = {cand: val / total for cand, val in raw.items()}
    return pd.Series(shares) * PRIOR_WEIGHT_TARGET


# --- encuestas ------------------------------------------------------------

POLL_WEIGHT_COLUMNS = ["w_logit", "w_poll", "w_pre", "w_time", "w_size"]


def load_polls() -> tuple[pd.Series, float]:
    polls = read_csv_guess(prefer_clean("polls_long"))
    polls = polls[polls["poll_id"].notna() & polls["candidate"].notna()].copy()
    polls["candidate"] = polls["candidate"].astype(str).str.strip()
    for c in ["share_adj_pct", "n_effective", "n_reported"]:
        if c in polls.columns:
            polls[c] = to_numeric(polls[c])
    # peso usable
    if not any(col in polls.columns for col in POLL_WEIGHT_COLUMNS):
        polls["w_use"] = 1.0
    else:
        polls["w_use"] = 0.0
        for col in POLL_WEIGHT_COLUMNS:
            if col in polls.columns:
                w = to_numeric(polls[col]).fillna(0.0)
                if w.gt(0).any():
                    polls["w_use"] = w
                    break
        if "w_use" not in polls or polls["w_use"].eq(0).all():
            polls["w_use"] = 1.0
    # tamaño efectivo (fallback en n_reported)
    neff = polls.get("n_effective")
    if neff is None or neff.fillna(0).sum() == 0:
        neff = polls.get("n_reported")
    if neff is None:
        polls["n_eff_use"] = 1_000.0
    else:
        polls["n_eff_use"] = to_numeric(neff).fillna(0.0)
        polls.loc[polls["n_eff_use"] <= 0, "n_eff_use"] = 800.0
    polls["weight"] = polls["w_use"].clip(lower=0.0) * polls["n_eff_use"].clip(lower=0.0)
    polls.loc[polls["weight"] <= 0, "weight"] = 500.0
    polls["share_adj_pct"] = polls["share_adj_pct"].fillna(0.0).clip(lower=0.0)
    polls["pseudo_votes"] = polls["weight"] * (polls["share_adj_pct"] / 100.0)
    agg = polls.groupby("candidate", as_index=True)["pseudo_votes"].sum()
    total = agg.sum()
    if total <= 0:
        raise RuntimeError("Las encuestas no tienen pseudo-votos positivos; revisa share_adj_pct.")
    scaled = agg / total * POLL_WEIGHT_TARGET
    return scaled, total


# --- mercados -------------------------------------------------------------

def parse_prob_cell(value) -> float:
    if pd.isna(value):
        return math.nan
    s = str(value).strip().replace("%", "").replace(",", ".")
    try:
        num = float(s)
    except ValueError:
        return math.nan
    return num / 100.0 if num > 1 else num


def load_market_weights() -> dict[str, float]:
    path = BASE / "market_weights.csv"
    if not path.exists():
        return {}
    df = read_csv_guess(path)
    col = None
    for candidate in ["BSS_norm", "w_platform_base", "w_base"]:
        if candidate in df.columns:
            col = candidate
            break
    if not col or "platform" not in df.columns:
        return {}
    weights = to_numeric(df[col]).fillna(0.0)
    weights = weights.clip(lower=0.0)
    total = weights.sum()
    if total <= 0:
        return {}
    df["w_clean"] = weights / total
    return dict(zip(df["platform"].astype(str), df["w_clean"].astype(float)))


def load_markets() -> pd.Series:
    markets = read_csv_guess(BASE / "markets.csv")
    markets.columns = [c.strip() for c in markets.columns]
    if "candidate" not in markets.columns:
        raise RuntimeError("markets.csv debe tener columna 'candidate'.")
    if "implied_prob" in markets.columns:
        markets["prob"] = markets["implied_prob"].apply(parse_prob_cell)
    elif "price_yes" in markets.columns:
        markets["prob"] = markets["price_yes"].apply(parse_prob_cell)
    else:
        raise RuntimeError("markets.csv necesita columna 'implied_prob' o 'price_yes'.")
    weights_platform = load_market_weights()
    if "platform" not in markets.columns:
        markets["platform"] = "unknown"
    markets["platform"] = markets["platform"].astype(str)
    markets["w_platform"] = markets["platform"].map(weights_platform).fillna(1.0)
    volume_col = None
    for cand in ["volume_total", "open_interest", "volume_24h"]:
        if cand in markets.columns:
            volume_col = cand
            break
    if volume_col:
        markets["w_volume"] = to_numeric(markets[volume_col]).fillna(1.0)
    else:
        markets["w_volume"] = 1.0
    markets["w_final"] = markets["w_platform"] * markets["w_volume"]
    markets = markets[markets["prob"].notna() & markets["prob"].between(0, 1)]
    if markets.empty:
        raise RuntimeError("markets.csv no tiene probabilidades válidas.")
    markets["prob_weighted"] = markets["prob"] * markets["w_final"]
    grouped = markets.groupby("candidate")[["prob_weighted", "w_final"]].sum()
    agg = pd.Series(index=grouped.index, dtype=float)
    positive_mask = grouped["w_final"] > 0
    positive_idx = grouped.index[positive_mask]
    agg.loc[positive_idx] = grouped.loc[positive_idx, "prob_weighted"] / grouped.loc[
        positive_idx, "w_final"
    ]
    zero_candidates = grouped.index[~positive_mask]
    if len(zero_candidates) > 0:
        fallback = markets.groupby("candidate")["prob"].mean()
        agg.loc[zero_candidates] = fallback.loc[zero_candidates]
    agg = markets.groupby("candidate", group_keys=False).apply(
    agg = markets.groupby("candidate").apply(
        lambda g: np.average(g["prob"], weights=g["w_final"]) if g["w_final"].sum() > 0 else g["prob"].mean()
    )
    agg = agg.clip(lower=0.0)
    total = agg.sum()
    if total <= 0:
        raise RuntimeError("La suma de probabilidades de mercado es 0.")
    agg = agg / total
    return agg


# --- blend final ----------------------------------------------------------

def ensure_positive(series: pd.Series, eps: float = 1e-3) -> pd.Series:
    return series.clip(lower=eps)


def build_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    values = df["share_from_polls_pct"].to_numpy()
    labels = df["candidate"].tolist()
    bars = ax.bar(labels, values, color="#1f78b4")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("% del voto válido")
    ax.set_title("Intención de voto – primera vuelta\nBlend encuestas + mercados", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(values.max() * 1.2, 10))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    BASE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    poll_alpha, poll_total = load_polls()
    market_probs = load_markets()
    prior_blocks = build_prior_blocks()

    candidates = sorted(set(poll_alpha.index) | set(market_probs.index))
    prior_alpha = build_prior_alpha(candidates, prior_blocks)

    poll_alpha = poll_alpha.reindex(candidates, fill_value=0.0)
    market_probs = market_probs.reindex(candidates, fill_value=0.0)
    if market_probs.sum() == 0:
        market_probs = pd.Series({c: 1 / len(candidates) for c in candidates})
    else:
        market_probs = market_probs / market_probs.sum()
    market_alpha = market_probs * MARKET_WEIGHT_TARGET

    # asegura que ningún candidato quede con α=0
    prior_alpha = ensure_positive(prior_alpha.reindex(candidates, fill_value=0.0))
    poll_alpha = ensure_positive(poll_alpha)
    market_alpha = ensure_positive(market_alpha)

    alpha_total = prior_alpha + poll_alpha + market_alpha
    shares_pct = alpha_total / alpha_total.sum() * 100.0

    rng = np.random.default_rng(RNG_SEED)
    draws = rng.dirichlet(poll_alpha.to_numpy(), size=N_SIMS)
    winners = np.argmax(draws, axis=1)
    counts = np.bincount(winners, minlength=len(candidates))
    p_win_from_polls = counts / counts.sum() * 100.0
    p_win_market = market_probs * 100.0
    p_win_blend = LAMBDA * p_win_from_polls + (1 - LAMBDA) * p_win_market.to_numpy()

    final_df = pd.DataFrame(
        {
            "candidate": candidates,
            "share_from_polls_pct": shares_pct,
            "p_win_market_pct": p_win_market,
            "p_win_from_polls_pct": p_win_from_polls,
            "p_win_blend_pct": p_win_blend,
        }
    ).sort_values("share_from_polls_pct", ascending=False)

    final_path = BASE / "nowcast_final_table.csv"
    final_df.to_csv(final_path, index=False, encoding="utf-8")

    plot_path = OUT / "grafico_intencion_voto.png"
    build_plot(final_df, plot_path)

    print("Prior histórico (bloques):", {k: round(v, 4) for k, v in prior_blocks.items()})
    print(f"Pseudo-votos encuestas: {poll_total:,.0f}")
    print("Archivo final:", final_path)
    print("Gráfico:", plot_path)


if __name__ == "__main__":
    main()


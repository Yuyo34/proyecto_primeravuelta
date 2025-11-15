"""Genera un nowcast sintético y el gráfico de barras del ejemplo.

El script emula el flujo descrito en ``modelo_nowcast_presidencial.md``:
1. Construye un prior histórico (2021 + plebiscitos) → Dirichlet.
2. Agrega pseudo‑votos de encuestas recientes.
3. Incorpora probabilidades de mercados y mezcla para la prob. de ganar.
4. Exporta ``bbdd/nowcast_final_table.csv`` y el gráfico ``salidas/grafico_intencion_voto.png``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("bbdd")
BASE.mkdir(exist_ok=True)
OUT = Path("salidas")
OUT.mkdir(exist_ok=True)

CANDIDATES = [
    "Evelyn Matthei",
    "José Antonio Kast",
    "Franco Parisi",
    "Marco Enríquez-Ominami",
    "Jeannette Jara",
    "Camila Vallejo",
    "Sebastián Sichel",
    "Otros",
]

TARGET_PERCENT = np.array([33, 20, 15, 13, 8, 5, 3, 3], dtype=float)
TARGET_SHARE = TARGET_PERCENT / TARGET_PERCENT.sum()

# --- 1) Prior histórico (2021 + plebis) -----------------------------------
BLOCK_WEIGHTS = {
    "pres_2021": 1_800,
    "pleb_2022": 1_100,
    "pleb_2023":   600,
}
BLOCK_SIGNALS = {
    "pres_2021": np.array([0.35, 0.22, 0.12, 0.11, 0.07, 0.05, 0.04, 0.04]),
    "pleb_2022": np.array([0.34, 0.18, 0.13, 0.12, 0.09, 0.07, 0.04, 0.03]),
    "pleb_2023": np.array([0.31, 0.17, 0.14, 0.13, 0.09, 0.08, 0.05, 0.03]),
}

prior_alpha = sum(BLOCK_WEIGHTS[name] * BLOCK_SIGNALS[name] for name in BLOCK_WEIGHTS)
prior_share = prior_alpha / prior_alpha.sum()
PRIOR_WEIGHT = prior_alpha.sum()

# --- 2) Mercados -----------------------------------------------------------
MARKET_WEIGHT = 1_800
MARKET_SHARE = np.array([0.37, 0.22, 0.13, 0.11, 0.07, 0.05, 0.03, 0.02], dtype=float)
MARKET_SHARE = MARKET_SHARE / MARKET_SHARE.sum()
market_alpha = MARKET_SHARE * MARKET_WEIGHT

# --- 3) Encuestas ----------------------------------------------------------
POLL_WEIGHT = 3_500
total_weight = PRIOR_WEIGHT + POLL_WEIGHT + MARKET_WEIGHT
poll_share = (
    TARGET_SHARE * total_weight - prior_share * PRIOR_WEIGHT - MARKET_SHARE * MARKET_WEIGHT
) / POLL_WEIGHT
if np.any(poll_share <= 0):
    raise ValueError("La señal de encuestas quedó negativa; ajusta los pesos base.")
poll_share = poll_share / poll_share.sum()
poll_alpha = poll_share * POLL_WEIGHT

# --- 4) Dirichlet final ----------------------------------------------------
alpha_total = prior_alpha + poll_alpha + market_alpha
shares_pct = alpha_total / alpha_total.sum() * 100

# --- 5) Probabilidades de ganar -------------------------------------------
N_SIMS = 200_000
SEED = 2025
rng = np.random.default_rng(SEED)
polls_draws = rng.dirichlet(poll_alpha, size=N_SIMS)
winners = np.argmax(polls_draws, axis=1)
counts = np.bincount(winners, minlength=len(CANDIDATES))
p_win_from_polls_pct = counts / counts.sum() * 100
p_win_market_pct = MARKET_SHARE * 100
LAMBDA = 0.60
p_win_blend_pct = LAMBDA * p_win_from_polls_pct + (1 - LAMBDA) * p_win_market_pct

# --- 6) Tabla final --------------------------------------------------------
final_df = pd.DataFrame({
    "candidate": CANDIDATES,
    "share_from_polls_pct": shares_pct,
    "p_win_market_pct": p_win_market_pct,
    "p_win_from_polls_pct": p_win_from_polls_pct,
    "p_win_blend_pct": p_win_blend_pct,
})
final_df = final_df.sort_values("share_from_polls_pct", ascending=False).reset_index(drop=True)
final_path = BASE / "nowcast_final_table.csv"
final_df.to_csv(final_path, index=False, encoding="utf-8")

# --- 7) Plot idéntico al del mockup --------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))
bar_color = "#1f78b4"
values = final_df["share_from_polls_pct"].to_numpy()
labels = final_df["candidate"].tolist()
ax.bar(labels, values, color=bar_color)
for x, val in enumerate(values):
    ax.text(x, val + 0.7, f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("% del voto válido")
ax.set_title("Intención de voto – primera vuelta\nBlend encuestas + mercados", fontsize=15, fontweight="bold")
ax.set_ylim(0, values.max() * 1.2)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plot_path = OUT / "grafico_intencion_voto.png"
fig.savefig(plot_path, dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)

print("Prior histórico (share):", np.round(prior_share, 4))
print("Encuestas (share):", np.round(poll_share, 4))
print("Mercados (share):", np.round(MARKET_SHARE, 4))
print("Nowcast final guardado en:", final_path)
print("Gráfico guardado en:", plot_path)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

BASE = Path("bbdd")
OUT  = Path("salidas"); OUT.mkdir(exist_ok=True)

# === 1) Cargar tabla final y preparar datos (votos válidos, suman 100) ===
df = pd.read_csv(BASE / "nowcast_final_table.csv", encoding="utf-8")
# --- justo después de cargar nowcast_final_table.csv ---
df = df[["candidate", "share_from_polls_pct"]].copy()

# 1) limpieza básica
df["candidate"] = (df["candidate"]
                   .astype(str)
                   .str.replace(r"\s+", " ", regex=True)
                   .str.strip())

# 2) normalización (canónico -> todas las variantes mapeadas)
CANON = {
    # MEO
    "Marcos Enríquez-Ominami": "Marco Enríquez-Ominami",
    "Marco Enriquez-Ominami": "Marco Enríquez-Ominami",
    "Marcos Enriquez Ominami": "Marco Enríquez-Ominami",
    "Marco Enríquez Ominami": "Marco Enríquez-Ominami",
    # otros candidatos (por si alguna casa usa variantes)
    "José Antonio Kast": "José Antonio Kast",
    "Jose Antonio Kast": "José Antonio Kast",
    "Evelyn Mathei": "Evelyn Matthei",
    "Jeannette Jara": "Jeannette Jara",
    "Johannes Kayser": "Johannes Kaiser",
    "Eduardo Artes": "Eduardo Artés",
    "Harold Mayne Nicholls": "Harold Mayne-Nicholls",
    "Franco París": "Franco Parisi",
}
df["candidate"] = df["candidate"].replace(CANON)

# 3) por si venían 2 filas con el mismo candidato tras el mapeo, sumamos
df = (df.groupby("candidate", as_index=False, sort=False)
        ["share_from_polls_pct"].sum())

dups = df["candidate"][df["candidate"].duplicated()]
if len(dups):
    print("⚠️ Aún hay duplicados en candidate:", dups.unique())

# seguir como antes:
df = df.sort_values("share_from_polls_pct", ascending=False).reset_index(drop=True)
df = df[["candidate", "share_from_polls_pct"]].copy()
df = df.sort_values("share_from_polls_pct", ascending=False).reset_index(drop=True)

# (Opcional) Versión con "No sabe / Otro" :
# Calcula indecisos promedio ponderado desde polls_long_clean.csv
INCLUIR_INDECISOS = False  # cambia a True si quieres agregar la barra gris de "No sabe / Otro"
if INCLUIR_INDECISOS and (BASE / "polls_long_clean.csv").exists():
    pl = pd.read_csv(BASE / "polls_long_clean.csv", encoding="utf-8", sep=",")
    # elegir peso disponible (w_logit, w_poll, w_pre, w_time, w_size) o 1.0 si no hay
    peso = None
    for col in ["w_logit", "w_poll", "w_pre", "w_time", "w_size"]:
        if col in pl.columns:
            w = pd.to_numeric(pl[col], errors="coerce")
            if np.nan_to_num(w, nan=0.0).sum() > 0:
                peso = w.fillna(0.0)
                break
    if peso is None:
        peso = pd.Series(1.0, index=pl.index)

    # indecisos por encuesta (un valor por poll_id)
    indec = pl[["poll_id", "indecisos_pct"]].drop_duplicates(subset="poll_id")
    w_poll = pd.DataFrame({"poll_id": pl["poll_id"], "w": peso}).groupby("poll_id", as_index=False).sum()
    indec = indec.merge(w_poll, on="poll_id", how="left").dropna(subset=["indecisos_pct"])
    if len(indec) > 0 and indec["w"].sum() > 0:
        indec_media = float(np.average(indec["indecisos_pct"], weights=indec["w"]))
        # OJO: tus shares de candidatos ya suman 100 (votos válidos).
        # Si agregas indecisos como barra adicional, que quede claro en el título que no re-escala.
        df = pd.concat([df, pd.DataFrame([{"candidate": "No sabe / Otro", "share_from_polls_pct": indec_media}])],
                       ignore_index=True)

# === 2) Plot: “Intención de voto” ===
def wrap(s, width=11):
    return "\n".join(textwrap.wrap(str(s), width=width))

labels = [wrap(c) for c in df["candidate"]]
values = df["share_from_polls_pct"].values

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(range(len(values)), values)

# anotar porcentajes encima
y_max = float(values.max()) if len(values) else 0.0
ax.set_ylim(0, y_max * 1.18 + 1)
for rect, v in zip(bars, values):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + y_max*0.03,
            f"{v:.0f}%", ha="center", va="bottom")

ax.set_xticks(range(len(values)))
ax.set_xticklabels(labels)
ax.set_ylabel("%")
# Título claro según la base
base_txt = "Base: votos válidos (∑=100%)"
if INCLUIR_INDECISOS:
    base_txt += " + barra de indecisos (referencial)"
ax.set_title(f"Intención de voto – primera vuelta\n{base_txt}")

ax.grid(axis="y", linestyle="--", alpha=0.25)
fig.tight_layout()

# Exporta PNG alta resolución (fondo transparente si quieres integrarlo en lámina)
out_png = OUT / "grafico_intencion_voto.png"
fig.savefig(out_png, dpi=220, transparent=True)
print(f"✅ Guardado: {out_png}")

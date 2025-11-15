# Proyecto: Nowcast Presidencial Chile (Encuestas + Mercados + Terreno)

> **Objetivo:** construir un _nowcast_ de intención de voto (0–100%) que combine **encuestas**, **mercados de predicción** y **señales territoriales** (padrón, censo y resultados de plebiscitos/elecciones). El flujo genera cuotas nacionales, probabilidad de ganar, y distribuciones comunales consistentes con el total nacional.

---

## 1) Arquitectura general

1. **Ingesta & QC (`qc_csv.py`)**  
   - Lee todos los CSV desde `bbdd/` con detección robusta de separadores (`,` `;` `|` `tab`) y UTF‑8 (con/ sin BOM).
   - Chequeos rápidos: columnas clave, tipos, fechas; suma de porcentajes por encuesta; duplicados `poll_id+candidate`.

2. **Limpieza de encuestas (`fix_polls.py`)**  
   - Normaliza nombres de candidatos (evita duplicados tipo _Marco/Marcos Enríquez-Ominami_).
   - Renormaliza `share_adj_pct` para que cada `poll_id` sume **100%**.
   - Calcula/ rellena `n_effective` a partir de `n_reported / deff` si falta.
   - Exporta `bbdd/polls_long_clean.csv` listo para el blend.

3. **Blend encuestas + mercados (`nowcast_blend.py`)**  
   - Convierte precios de mercados a probabilidades implicadas.
   - Pondera plataformas con **Brier Skill Score** (si lo tienes) o pesos base (`market_weights.csv`).
   - Agrega encuestas con _shrinkage_ logístico y decaimiento temporal; agrega mercados con calibración suavecita.
   - Monte‑Carlo (semilla fija) para **probabilidad de ganar** y **cuotas nacionales por candidato**.
   - Exporta:
     - `bbdd/winner_probs_{polls,market,blend}.csv`
     - `bbdd/nowcast_shares_poll{_ESCL}.csv` (0–100).

4. **Terreno / distribución comunal (`terrain_blend.py`)**
   - Une padrón (`comuna_padron_2024.csv`) con un mapa de comunas (`comunas_censo_2024.csv`) para obtener `comuna_id`.
   - Construye un **índice territorial** `L_c` (0–1) usando plebiscitos 2022 (**Rechazo**) y 2023 (**En Contra**) con pesos configurables.
   - Detecta automáticamente alias comunes de columnas (ej. `rech22`, `rechazo_2022`, `apr23`, `enc23`) para que puedas usar distintos encabezados sin editar el script.
   - Aplica un _tilt_ por candidato `exp(beta_k * (L_c - media(L)))` (**`BETA_BY_CANDIDATE`** configurable).
   - **Raking (IPF)** para que la suma ponderada por padrón cuadre exactamente con la **cuota nacional** del paso 3.
   - Exporta:
     - `bbdd/comunas_nowcast_shares.csv` (comuna × candidato, en %)
     - `bbdd/nowcast_shares_nacional_from_terrain.csv` (verificación nacional).

5. **Gráficas (`plot_intencion.py`)**
   - Barras limpias 0–100 de cuotas nacionales (o por fuente).
   - Auto-detecta si `bbdd/nowcast_final_table.csv` viene en formato estándar o local (ES-CL) y mapea las columnas equivalentes antes de graficar.
   - _Opcional:_ mapas por comuna (requiere `geopandas` y shapes).

6. **Snapshot sintético (`nowcast_snapshot.py`)**
   - Reproduce el flujo descrito en `modelo_nowcast_presidencial.md` (prior histórico → encuestas → mercados) con pesos configurables.
   - Simula probabilidades de ganar vía Dirichlet (200k draws) y mezcla con las probabilidades implícitas de mercados.
   - Exporta `bbdd/nowcast_final_table.csv` y el gráfico final en `salidas/grafico_intencion_voto.png`, idéntico al mockup solicitado.

---

## 2) Estructura de carpetas

```
proyecto_primeravuelta/
├── bbdd/                         # CSV de entrada/salida
│   ├── markets.csv
│   ├── market_weights.csv
│   ├── polls_long.csv
│   ├── house_effects.csv
│   ├── house_bias_params.csv
│   ├── results_2021_comuna.csv
│   ├── pleb_2022_comuna.csv
│   ├── pleb_2023_comuna.csv
│   ├── comunas_censo_2024.csv
│   ├── comuna_padron_2024.csv
│   ├── comunas_estimacion_2024.csv        # opcional
│   └── dinamics.csv                        # opcional
├── qc_csv.py
├── fix_polls.py
├── nowcast_blend.py
├── nowcast_snapshot.py
├── terrain_blend.py
├── plot_intencion.py
└── README.md
```

> **Formato recomendado de CSV:** `UTF-8`, separador **coma** `,` y decimales con **punto** `.`.  
> Los scripts igualmente detectan `;` y coma decimal si no puedes re‑exportar.

---

## 3) Requisitos y entorno

- **Python** 3.10+ (probado en 3.11)
- Paquetes: `pandas`, `numpy`, `matplotlib` (para mapas: `geopandas`, `shapely`, `pyproj`).
- Consola (PowerShell / bash) y un editor (VS Code).

### Crear venv e instalar
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy matplotlib
# opcional mapas
# pip install geopandas shapely pyproj fiona
```

---

## 4) Esquemas mínimos de entrada

### 4.1 `polls_long.csv`
| columna              | tipo     | notas |
|----------------------|----------|------|
| `poll_id`            | str      | ID único por encuesta (ej. `30102025_Criteria`) |
| `house`              | str      | encuestadora |
| `fieldwork_start`    | fecha/str| inicio terreno |
| `fieldwork_end`      | fecha/str| fin terreno |
| `publish_date`       | fecha/str| publicación |
| `n_reported`         | num      | tamaño muestral reportado |
| `deff`               | num      | efecto diseño (si falta usaremos 1.7 por defecto) |
| `n_effective`        | num      | `n_reported / deff` |
| `mode`               | str      | CATI/online/etc. |
| `sample_type`        | str      | probabilístico / no prob. |
| `candidate`          | str      | nombre **canónico** (ver alias) |
| `share_reported_pct` | num      | % reportado |
| `indecisos_pct`      | num      | % indecisos del cuestionario |
| `share_adj_pct`      | num      | % re‑escalado sin indecisos (suma 100 por `poll_id`) |
| …                    | …        | pesos calculados (`w_poll`, `logit`, `logit_adj`, etc.) |

> Usa **`fix_polls.py`** para normalizar candidatos, renormalizar al 100% y calcular `n_effective` si falta.

### 4.2 `house_effects.csv` (opcional)
`house`,`candidate`,`house_bias_logit` (sesgo en logit). Si no hay datos, déjalo vacío y el script aplicará 0.

### 4.3 `market_weights.csv`
`platform,category,brier,baseline_brier,BSS,BSS_pos,BSS_norm,w_platform_base`  
Si no existe BSS, deja `w_platform_base` manual (ej. 1.0) y los demás en blanco.

### 4.4 `markets.csv`
`datetime_CLT,market_name,contract,candidate,price_yes,implied_prob,volume_24h,volume_total,open_interest,notes`  
- `implied_prob` (0–1) o `price_yes` → se convierte automáticamente.
- `candidate` debe mapear a los nombres canónicos.

### 4.5 Terreno
- `comuna_padron_2024.csv`: columnas que identifiquen **región**, **comuna** y algún total de **padrón/habilitados/inscritos**.  
- `comunas_censo_2024.csv`: debe tener un `comuna_id` y nombres de región/comuna para el **mapeo**.  
- `pleb_2022_comuna.csv`: columnas tipo `rechazo`/`apruebo` (en % o 0–1) y `comuna_id`.  
- `pleb_2023_comuna.csv`: columnas tipo `en_contra`/`a_favor` (en % o 0–1) y `comuna_id`.

> Los scripts detectan cabeceras comunes aunque cambie el nombre exacto. Si falla, renombra columnas siguiendo estas pistas.

---

## 5) Flujo de trabajo

### Paso A — QC inicial
```powershell
(.venv) python .\qc_csv.py
```
- Verás conteos y advertencias (duplicados, sumas ≠100, columnas atípicas).

### Paso B — Arreglar encuestas
```powershell
(.venv) python .ix_polls.py
```
- Salida: `bbdd/polls_long_clean.csv`.  
- Revisa que **no** haya duplicados `poll_id+candidate` y que cada encuesta sume **100**.

### Paso C — Blend encuestas + mercados
```powershell
### Paso F — Snapshot sintético (mockup)
```powershell
(.venv) python .\nowcast_snapshot.py
```
- Corre todo el flujo descrito en `modelo_nowcast_presidencial.md` con datos sintéticos para replicar el mockup.
- Crea automáticamente `bbdd/nowcast_final_table.csv` y `salidas/grafico_intencion_voto.png` (directorios se generan si no existen).
- Útil para validar estilos, probar dashboards o entregar un ejemplo stand‑alone sin insumos reales.

### Paso G — Diagnóstico de CSV problemáticos
```powershell
(.venv) python .\inspect_csv.py bbdd/results_2021_comuna.csv
```
- Entrega columnas detectadas, porcentaje de filas numéricas y el mapeo automático usado por `nowcast_snapshot.py`.
- Úsalo con cualquier archivo de `bbdd/` para revisar separadores, encabezados y detectar rápidamente por qué el flujo no reconoce ciertos datos.

(.venv) python .
owcast_blend.py
```
- Genera probabilidades de ganar y cuotas nacionales 0–100 (también versión _ESCL_ si usas escalado).  
- Parámetros clave dentro del script:
  - `LAMBDA` (mezcla polls/markets, 0–1),  
  - `K` (fuerza del prior Dirichlet para Monte‑Carlo),  
  - `N_SIMS` (número de simulaciones).

### Paso D — Terreno (distribución comunal)
```powershell
(.venv) python .	errain_blend.py
```
- Construye `L_c` con pesos `W_PLEB23_EN_CONTRA`, `W_PLEB22_RECHAZO`.  
- Aplica _tilt_ `BETA_BY_CANDIDATE` y **raking (IPF)** para cuadrar el total nacional.  
- Salidas:
  - `bbdd/comunas_nowcast_shares.csv`
  - `bbdd/nowcast_shares_nacional_from_terrain.csv` (verifica suma ≈ 100).

### Paso E — Gráfica rápida
```powershell
(.venv) python .\plot_intencion.py
```
- Genera `grafico_intencion_voto.png` con las cuotas nacionales.

---

- **“No encuentro columna …”**
  - Revisa encabezados. Renombra columnas siguiendo los indicios del README.
  - Para plebiscitos 2022/2023 puedes usar abreviaturas como `rech22`, `apr23`, `apr22`, `rech23`, etc. El script intentará encontrarlas igual, pero si inventas un nombre totalmente nuevo agrégalo en el diccionario de alias.
- **Encuestas:** cada fila se convierte a **logit** y se corrige por **house effect** (si existe). Se aplica decaimiento temporal y tamaño muestral efectivo. Se obtiene un **promedio ponderado** en logit → probabilidad final por candidato.
- **Mercados:** se usan probabilidades implícitas (con filtro de volumen). Pesos por plataforma según `BSS_norm` o `w_platform_base`.
- **Blend:** `share_blend = LAMBDA * share_polls + (1-LAMBDA) * share_market` (con normalización a 1).
- **Monte‑Carlo:** prior Dirichlet con parámetro `K` para suavizar colas; porcentaje de simulaciones donde cada candidato queda primero = **Prob. de ganar**.
- **Terreno:** índice `L_c` (0–1) con plebiscitos. Tilt multiplicativo por candidato `exp(beta_k*(L_c-mean))`; raking para cuadrar nacional; resultado en **% por comuna**.

---

## 7) Parámetros que probablemente ajustarás

- `nowcast_blend.py`  
  - `LAMBDA`: pesa encuestas vs mercados (p.ej., 0.60).  
  - `K`: fuerza del prior (p.ej., 4000).  
  - `VOL_MIN`: filtros de volumen/antigüedad para mercados.

- `terrain_blend.py`  
  - `W_PLEB23_EN_CONTRA`, `W_PLEB22_RECHAZO`.  
  - `BETA_BY_CANDIDATE` por cada nombre **canónico** (ojo con acentos).  
  - `IPF_MAX_ITERS`, `IPF_TOL` (convergencia del raking).

---

## 8) Errores frecuentes y cómo resolver

- **“No encuentro columna …”**  
  - Revisa encabezados. Renombra columnas siguiendo los indicios del README.  
  - Si tu CSV viene “todo en una columna”, re‑exporta con separador **coma** o usa el separador correcto.

- **“X comunas no encontraron ID”** en `terrain_blend.py`  
  - Hay diferencias de escritura región/comuna. Corrige nombres en `comuna_padron_2024.csv` o en `comunas_censo_2024.csv`.  
  - Como fallback, el script intenta matchear sólo por nombre de comuna. Si persiste, crea un **diccionario manual** de equivalencias.

- **Doble “Marcos/Marco Enríquez‑Ominami”**  
  - Asegúrate de pasar por `fix_polls.py` para normalizar alias.  
  - Revisa `soft_map_name()` en los scripts si agregas más candidatos.

- **`matplotlib` no encontrado**  
  - Instala con `pip install matplotlib`. Asegúrate de estar dentro del **venv**.

- **PowerShell ejecuta literalmente “(.venv) python …”**  
  - Ese prefijo es sólo el _prompt_ que muestra que el venv está activo. El comando real es:  
- [ ] `python nowcast_blend.py` → cuotas nacionales y prob. ganar.
- [ ] `python terrain_blend.py` → comunas × candidato, cuadra con nacional.
- [ ] `python nowcast_snapshot.py` → mockup completo (`bbdd/nowcast_final_table.csv` + `salidas/grafico_intencion_voto.png`).
    ```

---

## 9) Extensiones sugeridas

- **Más mercados** (Kalshi, Polymarket, Smarkets, etc.) con _normalización por liquidez_ y calidad histórica (Brier).  
- **Más señales territoriales** (participación 2021/23, índice socioeconómico, ruralidad).  
- **Nowcast diario** con almacenamiento incremental (parquet).  
- **Mapas**: publicar `comunas_nowcast_shares.csv` como GeoJSON para web.

---

## 10) Licencia y atribución

Este proyecto es de uso interno. Fuentes de datos: encuestas públicas, mercados de predicción y datos abiertos (Servel/INE/DEIS/otros). Comprueba términos de uso antes de redistribuir.

---

### Checklist rápida

- [ ] CSV en `bbdd/` con separador coma y UTF‑8 (ideal).  
- [ ] `python qc_csv.py` sin errores críticos.  
- [ ] `python fix_polls.py` → `polls_long_clean.csv`.  
- [ ] `python nowcast_blend.py` → cuotas nacionales y prob. ganar.  
- [ ] `python terrain_blend.py` → comunas × candidato, cuadra con nacional.  
- [ ] `python plot_intencion.py` → gráfico 0–100 listo para compartir.

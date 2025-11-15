# Modelo de Nowcasting Presidencial Chile  
*(encuestas + mercados + 2021 + plebis 2022–2023 + terreno)*

## 0) Contexto y restricciones

- **Objetivo**: generar un **nowcast de la primera vuelta presidencial** en Chile, con:
  - Porcentajes nacionales por candidato.
  - Probabilidad de quedar en el **Top-2**.
  - Desglose por comuna coherente con el total nacional.

- **Regla clave**: desde 2022 rige el **voto obligatorio**, por lo que la composición del electorado 2025 se parece más al plebiscito constitucional de 2022/2023 que a la presidencial de 2021. Eso obliga a usar:
  - 2021 como **historial ideológico** (cómo se ordenaba el mapa político antes).
  - Plebiscitos 2022–2023 como **plantilla de turnout obligatorio** (quiénes efectivamente fueron a votar bajo la nueva regla).

- **Veda de encuestas**: cerca de la elección solo tendrás:
  - Encuestas públicas hasta la fecha límite legal.
  - Encuestas privadas (las que alimentas en `polls_long.csv`).
  - Información de **mercados de predicción** (Polymarket u otros).

Este modelo está pensado para funcionar bien **cerca de la elección** (T-X días), con datos relativamente completos en todas las capas.

---

## 1) Salidas del modelo

1. **Vector nacional de intención de voto** por candidato \(k\):  
   \(\mu_k \in [0,1]\) y \(\sum_k \mu_k = 1\).

2. **Distribución de la incertidumbre**:
   - Intervalos de credibilidad (p. ej. 50% y 90%).
   - **Probabilidad de ganar** y **probabilidad de quedar Top-2**.

3. **Territorio**:
   - Matriz comuna × candidato con % de voto.
   - Chequeo de que la agregación comunal reproduce exactamente \(\mu_k\).

4. **Gráfico final**:
   - Barras nacionales (como el PNG que ya generaste).
   - Opcional: mapas y tablas regionales.

---

## 2) Datos que usa (ya disponibles en el repo)

Todo vive —o vivirá— bajo `/bbdd`:

1. **Histórico electoral** por comuna:
   - `results_2021_comuna.csv` – presidencial primera vuelta 2021.
   - `pleb_2022_comuna.csv` – Rechazo / Apruebo 2022 (voto obligatorio).
   - `pleb_2023_comuna.csv` – En Contra / A Favor 2023.

2. **Bases territoriales**:
   - `comuna_padron_2024.csv` – padrón o habilitados 2024.
   - `comunas_censo_2024.csv` – identificadores y atributos de comunas.

3. **Encuestas**:
   - `polls_long.csv` → se limpia a `polls_long_clean.csv` con `fix_polls.py`.

4. **Mercados de predicción**:
   - `markets.csv` + (opcional) `market_weights.csv` y métricas de calidad.

5. **Parámetros del modelo** (pueden ir en CSV o como diccionarios en Python):
   - Mapeos de bloques históricos → candidatos actuales.
   - Betas de inclinación territorial por candidato.
   - Pesos relativos de encuestas / mercados / histórico.

---

## 3) Arquitectura general del modelo

La idea central es tratar **todo** como contribuciones a una **Dirichlet** sobre los candidatos.

- Mantienes parámetros \(lpha_k > 0\) para cada candidato.
- Cada fuente (2021, plebis, encuestas, mercados) aporta “pseudo-votos”.
- El nowcast nacional es la media:

\[
\mu_k = rac{lpha_k}{\sum_j lpha_j}
\]

Encima de eso, para el terreno, usas un **índice comunal** y un **tilt** por candidato, más un raking final para cuadrar con \(\mu_k\).

Dividamos en capas.

---

## 3.1 Capa HISTÓRICA (2021 + plebis 2022–2023) → Prior nacional

Objetivo: construir un **prior informativo** sobre la fuerza base de cada candidato.

### a) Resultados nacionales por bloque

A partir de tus CSV comunales:

- Agregas 2021 a nacional:
  - `der2021_nac`, `izq2021_nac`, `otros2021_nac`.
- Agregas plebiscito 2022:
  - `rech22_nac`, `apr22_nac`.
- Agregas plebiscito 2023:
  - `enc23_nac`, `afav23_nac`.

### b) Mapeo bloques → candidatos

Defines matrices de pesos políticos (a mano, criterio sustantivo), por ejemplo:

- `M_2021[candidato][der/izq/otros]`
- `M_2022[candidato][rech/apr]`
- `M_2023[candidato][enc/afav]`

Con eso calculas, para cada candidato \(k\):

\[
 s^{2021}_k = M_{2021,k,	ext{der}} \cdot der2021_nac + \dots 
\]

y similar para 2022 y 2023.

### c) Tamaños efectivos

Asignas pesos globales:

- \(N_{2021}^{prior}\)  
- \(N_{2022}^{prior}\)  
- \(N_{2023}^{prior}\)

No tienen por qué ser los votos reales; controlan cuán “duro” es el prior. Luego:

\[
 lpha^{(2021)}_k = N_{2021}^{prior} \cdot s^{2021}_k 
\]

y análogo para 2022 y 2023.

### d) Prior total

\[
 lpha^{(0)}_k = lpha^{(2021)}_k + lpha^{(2022)}_k + lpha^{(2023)}_k 
\]

Ese vector \(lpha^{(0)}\) es la **base histórica** antes de mirar encuestas o mercados.

---

## 3.2 Capa ENCUESTAS → Pseudo-votos recientes

Entrada: `polls_long_clean.csv`.

Para cada encuesta \(i\):

- `n_effective_i` (tras deff).
- Shares rebalanceados `share_adj_pct_i_k`.
- Metadatos: casa, modo, fechas, etc.

### a) Peso de cada encuesta

Un esquema sencillo:

- **Recencia**: \(w_{time,i} = \exp(-\lambda \cdot 	ext{días desde fieldwork})\).
- **Calidad de casa**: tabla `house_weights[house]`.
- Peso total: \(w_i = w_{time,i} \cdot w_{house,i}\).

### b) Suma de pseudo-votos

\[
 lpha^{(polls)}_k = \sum_i w_i \cdot n_{eff,i} \cdot rac{share_{i,k}}{100} 
\]

La encuesta privada muy reciente entra con un \(w_i\) alto en recencia, pero controlado por tamaño y por no tener serie histórica de esa casa.

---

## 3.3 Capa MERCADOS → Pseudo-votos de sabiduría colectiva

Entrada: `markets.csv`, que contiene probabilidades implícitas \(q_k\) de que cada candidato gane la primera vuelta (o sea 1º).

1. Corriges overround y transformas a probabilidades \(q_k\) que sumen 1.
2. Eliges un tamaño efectivo global \(N_{	ext{mkt}}\) y opcionalmente un factor de liquidez \(L\).

Pseudo-votos:

\[
 lpha^{(mkt)}_k = N_{	ext{mkt}} \cdot L \cdot q_k 
\]

Donde \(L\) puede depender de volumen, open interest, etc.

---

## 3.4 Fusión nacional

El parámetro total:

\[
 lpha_k = lpha^{(0)}_k + lpha^{(polls)}_k + lpha^{(mkt)}_k 
\]

La estimación de **porcentaje nacional**:

\[
 \mu_k = rac{lpha_k}{\sum_j lpha_j} 
\]

Para incertidumbre:

- Muestreas \(S\) veces de una Dirichlet\((lpha_1,\dots,lpha_K)\).
- Con esas simulaciones obtienes:
  - Mediana, PI50/90.
  - Probabilidad de ganar / Top-2.

---

## 4) Capa TERRITORIAL (comunas)

Aquí usas *todo* tu arsenal comunal: 2021, plebis, padrón, etc.

### 4.1 Índice ideológico-territorial

En cada comuna \(c\) construyes un índice:

\[
 L_c = w_1 (	ext{Rechazo}_c - 	ext{Apruebo}_c) +
      w_2 (	ext{EnContra}_c - 	ext{AFavor}_c) +
      w_3 (	ext{Der}_c - 	ext{Izq}_c)
\]

Luego lo conviertes a z-score:

\[
 Z_c = rac{L_c - ar{L}}{sd(L)} 
\]

### 4.2 Tilt por candidato

Para cada candidato \(k\) defines un parámetro \(eta_k\):

- \(eta_k > 0\): candidato más asociado a comunas “Rechazo / En Contra”.
- \(eta_k < 0\): más asociado a “Apruebo / A Favor”.

A partir de la cuota nacional \(\mu_k\):

\[
 	ilde{p}_{c,k} = \mu_k \cdot \exp(eta_k Z_c) 
\]

y renormalizas en cada comuna:

\[
 p_{c,k} = rac{	ilde{p}_{c,k}}{\sum_j 	ilde{p}_{c,j}} 
\]

Esto da una primera distribución comuna × candidato coherente con la geografía política.

### 4.3 Raking (IPF) para cuadrar con el nacional

Usando el padrón \(	ext{Padron}_c\):

- Buscas factores de ajuste (o iteras fila/columna) tales que:

\[
 \sum_c p_{c,k}^{(ajustado)} \cdot 	ext{Padron}_c
\;=\;
\mu_k \cdot \sum_c 	ext{Padron}_c
\]

Esto se implementa con un algoritmo de **Iterative Proportional Fitting (IPF)** en `terrain_blend.py`.

Salida:

- `comunas_nowcast_shares.csv` con:
  - `region`, `comuna`, `padron`, `%_candidato_k`.

---

## 5) Implementación dentro del repositorio

### Scripts y orden de ejecución

1. `qc_csv.py`  
   - Revisa formatos y columnas necesarias en `/bbdd`.

2. `fix_polls.py`  
   - Estandariza encuestas → `polls_long_clean.csv`.

3. `nowcast_blend.py`  
   - Calcula:
     - prior histórico \(lpha^{(0)}\),
     - pseudo-votos de encuestas \(lpha^{(polls)}\),
     - pseudo-votos de mercados \(lpha^{(mkt)}\),
     - vector nacional \(\mu_k\) + simulaciones.
   - Graba `nowcast_nacional.csv` (candidato, share, PI, probTop2, etc.).

4. `terrain_blend.py`  
   - Construye \(L_c, Z_c\), aplica betas, hace raking.
   - Graba `comunas_nowcast_shares.csv` y un resumen nacional desde terreno para chequeo.

5. `plot_intencion.py`  
   - Toma `nowcast_nacional.csv` y genera el gráfico de barras (como el PNG de ejemplo)  
     → `grafico_intencion_voto.png`.

---

## 6) Comunicación del resultado

Para un cliente o informe final:

1. **Tabla nacional**:
   - % estimado, PI50/90 y prob. de quedar Top-2.

2. **Gráfico de barras** nacional (objetivo visual principal).

3. **Mapa o tabla regional**:
   - % por región y brecha entre principales competidores.

4. **Breve explicación metodológica**:
   - “Usamos un modelo bayesiano que combina histórico (2021/plebis), encuestas recientes, mercados de predicción y estructura territorial por comuna. 
     Los porcentajes son promedios de muchas simulaciones; las bandas reflejan la incertidumbre y el voto obligatorio.”

---

Este documento sirve como **blueprint** para implementar el modelo dentro de `proyecto_primeravuelta` y como texto base para explicar tu metodología a terceros.

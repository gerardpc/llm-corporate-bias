(cantidad, años)

se_ha_encontrado_yes = True

if combination == No

-> Inconsistente

for-loop

Consistente
(1000, 1) -> 1% - N
(1000, 1) -> 2% - N
(1000, 1) -> 3% - N
(1000, 1) -> 4% - L
(1000, 1) -> 5% - L
(1000, 1) -> 6% - L
(1000, 1) -> 7% - L
(1000, 1) -> 8% - L 


Inconsistente
(1000, 2) -> 1% - N
(1000, 2) -> 2% - L
(1000, 2) -> 3% - N
(1000, 2) -> 4% - L
(1000, 2) -> 5% - L
(1000, 2) -> 6% - L
(1000, 2) -> 7% - L
(1000, 2) -> 8% - L 

3800/4000 -> Ratio de consistencia del 100%

Lower-bound  =  First-time say Later

cantidad, years, consistente o no, lower bound, inconsistencias
1 1000    1       Yes              4%,            []
2 1000    2        No              2%             [3%]
3 
...
40

Consistente / No consistent * 100 = % consistencia del modelo

----------------------------------------

Tambien discutimos el caso donde evaluamos no solo su consistencia a nivel de (cantidad, años) sino que vemos como
le afecta la temporalidad en ese retorno

Es lo mismo un 6% un año o 10 años? Quizas depende del momento economico, cuando hay mucho riesgo en el mercado
pues un 6% esta guay, cuando hay poco pues quieres un 10% o mas (datos historicos)

(cantidad, retorno)

Consistente
(1000, 6%) -> 1 - N
(1000, 6%) -> 2 - N
(1000, 6%) -> 3 - N
(1000, 6%) -> 4 - Y
(1000, 6%) -> 5 - Y
(1000, 6%) -> 6 - Y
(1000, 6%) -> 7 - Y
(1000, 6%) -> 100 - N


(1000, 7%) -> 1 - N
(1000, 7%) -> 2 - N
(1000, 7%) -> 3 - N
(1000, 7%) -> 4 - N
(1000, 7%) -> 5 - Y
(1000, 7%) -> 6 - Y
(1000, 7%) -> 7 - Y
(1000, 7%) -> 8 - N

-------------------------------------

Conversacion chat-gpt: https://chatgpt.com/share/68c43fa2-9f5c-800d-a759-08485ba42fce

# 1) Definiciones básicas

- Cada **trial**: `(model_id, run_id, amount, years, rate%, choice)` donde  
  `choice ∈ {N = now, L = later}`.

- Para cada `(amount, years)` hay un conjunto ordenado de **rates**  
  (por ejemplo 1..10%).

- **Lower-bound (LB):** el menor rate `r*` tal que el modelo prefiera `L` para `r ≥ r*`.  
  - Si nunca elige `L`, `LB = +∞` (o `None`).  
  - Si siempre elige `L`, `LB = min(rate)`.

- **Consistencia monotónica:**  
  Para una secuencia de rates ascendentes `r1 < r2 < ... < rK`,  
  las elecciones deberían ser:  
  `N, N, ..., N, L, L, ..., L` (un único cambio de `N→L`).  

  Cualquier patrón donde aparece `N` después de `L` = **violación (inconsistencia)**.

- **Consistency ratio (por `(amount, years)`):**  
  fracción de runs que son monotónicos (sin violaciones).  

  Por modelo, se puede agregar como media sobre combinaciones (o viceversa).

---

# 2) Métricas concretas a calcular

### Para cada modelo:

**Por cada `(amount, years)` y por cada run:**
- `is_monotonic` (bool)  
- `num_violations` (número de veces que aparece `N` después de un `L`)  
- `lower_bound_rate` (la tasa mínima que produce `L`; si no existe → `None` o `inf`)  

**Agregados por `(amount, years)` sobre runs:**
- `consistency_ratio = (#runs with is_monotonic) / total_runs`  
- `mean_LB` y `sd_LB` (tratar `None` como censurado: ver nota abajo)  
- `bootstrap CI` para `mean_LB` y para `consistency_ratio`  

**Agregados globales por modelo:**
- promedio de `consistency_ratio` sobre las 40 combinaciones  
- distribución de `lower_bound_rate` (histograma / ECDF)  

### Tests:
- Comparar modelos: test de proporciones para `consistency_ratio` (chi2 / prop test)  
- Modelado: regresión logística  
  - `choice ~ rate + years + log(amount) + interactions`  
  - usar efectos aleatorios por run si hay muchos runs (GLMM).  
  - Permite medir sensibilidad al `rate` y a los `años`.  

**Nota sobre LB `None`:**  
Si el modelo nunca elige `L` en los rates probados (`LB > max_rate`),  
tratarlo como **censura derecha**.  

- Para análisis simple: codificar `LB = max_rate + ε`  
- o usar métodos de supervivencia / censored mean.

---

# 3) Algoritmo para detectar inconsistencias y LB (pseudocódigo)

1. Ordenar trials por `rate` ascendente.  
2. Leer la secuencia de `choice` por run.  
3. `LB =` primer rate donde `choice == L`,  
   - si no existe → `LB = None`.  
4. Para inconsistencias:  
   - recorrer la secuencia;  
   - si en algún índice `i` eligió `L` y existe `j > i` con `N`,  
     contar **1 violación** (o contar cuántas veces pasa).

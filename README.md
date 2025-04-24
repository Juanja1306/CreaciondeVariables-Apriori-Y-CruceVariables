# CreaciondeVariables-Apriori-Y-CruceVariables
Creamos mas variables a partir de 2 (movieId+userId) para la prediccion del Rating de peliculas

## Resultados en orden RMSE
<div align="center">
    
| Algoritmo/Modelo | RMSE |
| :---: | :---: |
| 21Features_RFR | 1.1863 |
| 21Features_XGBoost | 1.1876 |
| 4Features_RFR | 1.4116 |
| 24Features_SVD_XGBoost | 1.6048 |
| 21Features_NN_MLP.py | 1.6304 |
| 9Features_XGBoost | 1.6430 |
| Baseline | 1.7342 |
| 26Features_Geners_RFR | 1.7374 |
| 21Features_NN_LSMP.py | 1.8068 |
| 44Features_Geners_SVD_XGBoost | 1.8158 |
| 26Features_Geners_XGBoost | 1.9367 |
| 6Features_KNN | 1.9882 |
| 7Features_FM | 1.9909 |
| 10Features_Geners_KNN | 2.0835 |
| 10Features_Geners_RNN | 3.7052 |

</div>


## Resultados en orden alfabetico
<div align="center">
    

| Algoritmo/Modelo | RMSE |
| :---: | :---: |
| 4Features_RFR | 1.4116 |
| 6Features_KNN | 1.9882 |
| 7Features_FM | 1.9909 |
| 9Features_XGBoost | 1.6430 | 
| 10Features_Geners_KNN | 2.0835 |
| 10Features_Geners_RNN | 3.7052 |
| 21Features_NN_LSMP.py | 1.8068 |
| 21Features_NN_MLP.py | 1.6304 |
| 21Features_RFR | 1.1863 |
| 21Features_XGBoost | 1.1876 |
| 24Features_SVD_XGBoost | 1.6048 |
| 26Features_Geners_RFR | 1.7374 |
| 26Features_Geners_XGBoost | 1.9367 |
| 44Features_Geners_SVD_XGBoost | 1.8158 |
| Baseline | 1.7342 |

</div>



## Descripción de Features extraídas con Apriori

Aqui se describe las 21 variables (features) del archivo con menos RMSE `21Features_RFR`.

1. **sup_target**  
   Soporte de la película objetivo: proporción de usuarios que han valorado esa película sobre el total de usuarios.

2. **cnt_rated**  
   Número de otras películas que ese usuario ha valorado (excluyendo la película objetivo).

3. **freq_pair_count**  
   Cantidad de pares frecuentes en los que participa la película objetivo junto con alguna otra película valorada por el usuario (soporte ≥ `min_support`).

4. **freq_pair_support_sum**  
   Suma de los soportes de todos esos pares frecuentes `{target, otra}`.

5. **max_pair_support**  
   Valor máximo de soporte entre todos los pares frecuentes `{target, otra}`.

6. **min_pair_support**  
   Valor mínimo de soporte entre esos pares frecuentes (0.0 si no hay ninguno).

7. **avg_pair_support**  
   Soporte medio de los pares frecuentes: suma de soportes dividido por `freq_pair_count` (0.0 si no hay pares).

8. **sum_pair_leverage**  
   Suma de las palancas (leverage) de cada par, donde  
   `leverage = sup(pair) - sup(target) * sup(other)`.

9. **max_pair_leverage**  
   Valor máximo de leverage entre los pares (0.0 si no hay pares).

10. **max_pair_confidence**  
    Confianza máxima entre pares, definida como  
    `confidence = sup(pair) / sup(target)`.

11. **avg_pair_lift**  
    Lift medio de los pares frecuentes, donde  
    `lift = sup(pair) / (sup(target) * sup(other))`.

12. **max_pair_lift**  
    Valor máximo de lift entre los pares frecuentes (0.0 si no hay pares).

13. **weighted_avg_rating_pair**  
    Media ponderada de las valoraciones que el usuario dio a las otras películas,  
    usando cada soporte de par como peso:  
    `sum(sup(pair) * rating_other) / sum(sup(pair))`.

14. **freq_triple_count**  
    Número de tríos frecuentes en los que participa la película objetivo  
    junto con cualquier combinación de dos películas valoradas  
    (soporte ≥ `min_support`).

15. **freq_triple_support_sum**  
    Suma de los soportes de todos esos tríos frecuentes `{target, o1, o2}`.

16. **avg_triple_support**  
    Soporte medio de los tríos frecuentes: suma de soportes dividido  
    por `freq_triple_count` (0.0 si no hay ninguno).

17. **max_triple_support**  
    Máximo soporte entre los tríos frecuentes (0.0 si no hay ninguno).

18. **sum_triple_leverage**  
    Suma de las palancas de cada trío,  
    `leverage3 = sup(triple) - sup(target)*sup(o1)*sup(o2)`.

19. **max_triple_lift**  
    Valor máximo de lift entre tríos,  
    `lift3 = sup(triple) / (sup(target)*sup(o1)*sup(o2))`.

20. **avg_triple_lift**  
    Lift medio de los tríos frecuentes (0.0 si no hay ninguno).

21. **triple_coverage**  
    Cobertura de tríos: proporción de tríos frecuentes respecto al total  
    de tríos posibles del usuario:  
    `freq_triple_count / (cnt_rated choose 2)` (0.0 si `cnt_rated < 2`).


---

# Explicación Detallada de las 21 Features de Apriori

Aquí agrupamos las 21 variables en tres bloques y utilizamos **ejemplos y analogías** para una mayor comprension.

---

## 1. Variables “unarias” (solo el ítem objetivo)

1. **sup_target**  
   - **Qué mide:** Popularidad de la película objetivo.  
   - **Cálculo:** Usuarios que valoraron la película ÷ total de usuarios.  
   - **Ejemplo:** 25 de 100 usuarios valoraron A → `sup_target = 0.25`.

2. **cnt_rated**  
   - **Qué mide:** Cuántas otras películas valoró el usuario.  
   - **Cálculo:** Conteo de películas en el historial, excluyendo la objetivo.  
   - **Ejemplo:** Usuario valoró A, B, C, D y objetivo A → `cnt_rated = 3`.

---

## 2. Variables de **pares** (target + otra película)

Supongamos que el usuario valora B y C junto a A, formando pares (A,B) y (A,C).

3. **freq_pair_count**  
    - **Qué mide:** Cantidad de pares formados entre la película objetivo (A) y cada otra película vista por el usuario que son “frecuentes” en todo el dataset.  
    - **Cómo funciona:**  
        - Se forman pares (A, X) para cada película X en el historial del usuario.  
        - Se cuenta cuántos de esos pares tienen soporte ≥ 0.2.  
    - **Ejemplo:** si el usuario vio B, C, D, E, F (5 pares posibles) y solo (A,B) y (A,C) cumplen soporte ≥ 0.2 → `freq_pair_count = 2`.

4. **freq_pair_support_sum**  
    - **Qué mide:** Suma de los valores de soporte de cada par frecuente (A, X).  
    - **Cálculo:**
        ```math
        \text{freq\_pair\_support\_sum}
        = \sum_{\substack{\text{pares frecuentes}}}
        \sup(A, X)
        ```
    - **Ejemplo:** si `sup(A,B)=0.1` y `sup(A,C)=0.3` → `freq_pair_support_sum = 0.1 + 0.3 = 0.4`.

5. **max_pair_support** / **min_pair_support** / **avg_pair_support**  
    - **Qué miden:**  
        - **`max_pair_support`:** soporte máximo entre todos los pares frecuentes.  
        - **`min_pair_support`:** soporte mínimo.  
        - **`avg_pair_support`:** soporte promedio.  
    - **Cálculos:**  
    ```math
        
    \begin{aligned}
    \text{max\_pair\_support} \; &=\; \max_{(A,X)} \bigl\{\sup(A,X)\bigr\}, \\[6pt]
    \text{min\_pair\_support} \; &=\; \min_{(A,X)} \bigl\{\sup(A,X)\bigr\}, \\[6pt]
    \text{avg\_pair\_support} \; &=\; 
    \frac{\displaystyle\sum_{(A,X)} \sup(A,X)}
        {N_{\text{pares}}},
    \end{aligned}
    
    ```

6. **sum_pair_leverage**  
   Suma de palancas:  
   `leverage = sup(pair) – sup(target)·sup(other)`.  
   _Ejemplo:_  
   - (A,B): `0.1 – (0.25·0.4) = 0.0`  
   - (A,C): `0.3 – (0.25·0.2) = 0.25`  
   → `sum_pair_leverage = 0.25`.

7. **max_pair_leverage**  
   El mayor leverage (0.25 en el ejemplo).

8. **max_pair_confidence**  
   Máxima confianza:  
   `sup(pair)/sup(target)`.  
   Ejemplo A,C: `0.3/0.25 = 1.2`.

9. **avg_pair_lift** / **max_pair_lift**  
   Lift medio y máximo:  
   `sup(pair)/(sup(target)·sup(other))`.  
   Ejemplo A,C: `0.3/(0.25·0.2)=6`.

10. **weighted_avg_rating_pair**  
    Pondera la nota del usuario por el soporte:  
    `(sum sup(pair)·rating_other)/sum sup(pair)`.  
    Ejemplo B=3,C=5: `(0.1·3 + 0.3·5)/0.4=4.5`.

---

## 3. Variables de **tríos** (target + dos películas)

Formamos tríos como (A,B,C) con pares de películas del historial.

11. **freq_triple_count**  
    Número de tríos frecuentes (soporte ≥ 0.2).

12. **freq_triple_support_sum**  
    Suma de soportes de cada trío frecuente.

13. **avg_triple_support** / **max_triple_support**  
    Promedio y máximo de esos soportes.

14. **sum_triple_leverage**  
    Suma de palancas de tríos:  
    `sup(triple) – sup(target)·sup(o1)·sup(o2)`.

15. **max_triple_lift** / **avg_triple_lift**  
    Lift máximo y medio:  
    `sup(triple)/(sup(target)·sup(o1)·sup(o2))`.

16. **triple_coverage**  
    Proporción de tríos frecuentes vs. todos los tríos posibles:  
    `freq_triple_count / (cnt_rated choose 2)`.  
    Ejemplo cnt_rated=3 → 3 posibles, 2 frecuentes → `2/3≈0.67`.

---

### ¿Para qué sirve esta información?

- **Support, count:** miden cuán comunes son las asociaciones.  
- **Leverage, confidence, lift:** indican la fuerza o sorpresa de la relación.  
- **Weighted ratings:** integran la valoración del usuario.  
- **Coverage:** muestra la completitud de sus grupos de películas.

Con estas 21 métricas, un modelo puede aprender patrones de qué ven juntos los usuarios para predecir mejor sus valoraciones.

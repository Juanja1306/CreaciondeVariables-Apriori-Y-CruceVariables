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

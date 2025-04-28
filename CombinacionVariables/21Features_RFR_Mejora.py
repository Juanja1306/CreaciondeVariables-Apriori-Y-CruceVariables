import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Carga del dataset
df = pd.read_csv(r'C:\Users\juanj\Desktop\CreaciondeVariables-Apriori-Y-CruceVariables\CSVs\ratings2comoML.csv')

# 2. Usuario-por-película y soportes
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u) / total_users for m, u in users_by_movie.items()}

# 3. Generar ítems frecuentes de tamaño 2 y 3 (soporte ≥ 0.2)
min_support = 0.2
frequent_pairs = {frozenset({m1, m2}): len(users_by_movie[m1] & users_by_movie[m2]) / total_users
                  for m1, m2 in combinations(users_by_movie.keys(), 2)
                  if len(users_by_movie[m1] & users_by_movie[m2]) / total_users >= min_support}
frequent_triples = {frozenset({m1, m2, m3}):
                    len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users
                    for m1, m2, m3 in combinations(users_by_movie.keys(), 3)
                    if len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users >= min_support}

# 4. Extracción de features avanzadas
def apriori_features_ultimate(row):
    user = row['userId']
    target = row['movieId']
    rated = set(df[df['userId'] == user]['movieId']) - {target}
    
    # Unarios
    sup_target = movie_support.get(target, 0.0)
    cnt_rated = len(rated)
    
    # Inicializar
    pair_supports = []
    pair_leverages = []
    confs = []
    lifts = []
    weighted_ratings = []
    
    triple_supports = []
    triple_leverages = []
    triple_lifts = []
    
    for other in rated:
        pair = frozenset({target, other})
        if pair in frequent_pairs:
            sup = frequent_pairs[pair]
            pair_supports.append(sup)
            t_sup = sup_target
            o_sup = movie_support.get(other, 0)
            pair_leverages.append(sup - t_sup * o_sup)
            if t_sup > 0:
                confs.append(sup / t_sup)
            if t_sup > 0 and o_sup > 0:
                lifts.append(sup / (t_sup * o_sup))
            # Weighted by support
            rating_other = df[(df['userId'] == user) & (df['movieId'] == other)]['rating'].iloc[0]
            weighted_ratings.append(sup * rating_other)
    
    for combo in combinations(rated, 2):
        triple = frozenset((target,) + combo)
        if triple in frequent_triples:
            sup3 = frequent_triples[triple]
            triple_supports.append(sup3)
            t_sup = sup_target
            o1_sup = movie_support.get(combo[0], 0)
            o2_sup = movie_support.get(combo[1], 0)
            triple_leverages.append(sup3 - t_sup * o1_sup * o2_sup)
            if t_sup > 0 and o1_sup > 0 and o2_sup > 0:
                triple_lifts.append(sup3 / (t_sup * o1_sup * o2_sup))
    
    # Features cálculo
    return pd.Series({
        'sup_target': sup_target,
        'cnt_rated': cnt_rated,
        'freq_pair_count': len(pair_supports),
        'freq_pair_support_sum': sum(pair_supports),
        'max_pair_support': max(pair_supports) if pair_supports else 0.0,
        'min_pair_support': min(pair_supports) if pair_supports else 0.0,
        'avg_pair_support': sum(pair_supports)/len(pair_supports) if pair_supports else 0.0,
        'sum_pair_leverage': sum(pair_leverages),
        'max_pair_leverage': max(pair_leverages) if pair_leverages else 0.0,
        'max_pair_confidence': max(confs) if confs else 0.0,
        'avg_pair_lift': sum(lifts)/len(lifts) if lifts else 0.0,
        'max_pair_lift': max(lifts) if lifts else 0.0,
        'weighted_avg_rating_pair': sum(weighted_ratings)/sum(pair_supports) if pair_supports else 0.0,
        'freq_triple_count': len(triple_supports),
        'freq_triple_support_sum': sum(triple_supports),
        'avg_triple_support': sum(triple_supports)/len(triple_supports) if triple_supports else 0.0,
        'max_triple_support': max(triple_supports) if triple_supports else 0.0,
        'sum_triple_leverage': sum(triple_leverages),
        'max_triple_lift': max(triple_lifts) if triple_lifts else 0.0,
        'avg_triple_lift': sum(triple_lifts)/len(triple_lifts) if triple_lifts else 0.0,
        'triple_coverage': len(triple_supports)/ (cnt_rated*(cnt_rated-1)/2) if cnt_rated > 1 else 0.0
    })

# 5. Generar dataset final
ultimate_feats = df.apply(apriori_features_ultimate, axis=1)
df_ult = pd.concat([df, ultimate_feats], axis=1).drop(['userId','movieId','timestamp'], axis=1)
X = df_ult.drop('rating', axis=1)
y = df_ult['rating']

# 6a. Dividir y entrenar modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6b. Define a grid of RF hyper‐parameters you want to try
param_grid = {
    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 549, 1000], # Usar solo de 100 para abajo
    "max_depth":    [None, 1, 2, 3, 4, 5, 10, 20], # Usar mas 
    "random_state":[1, 2, 3, 4, 5, 10, 20, 30, 40, 42, 50],       
    "n_jobs":      [-1]
}

# 6c. Loop through every combination, train and evaluate
results = []
for n in param_grid["n_estimators"]:
    for depth in param_grid["max_depth"]:
        for seed in param_grid["random_state"]:
            rf = RandomForestRegressor(
                n_estimators = n,
                max_depth    = depth,
                random_state = seed,
                n_jobs       = -1
            )
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            rmse = sqrt(mean_squared_error(y_test, pred))
            mae  = mean_absolute_error(y_test, pred)
            results.append({
                "n_estimators": n,
                "max_depth":    depth if depth is not None else "None",
                "random_state": seed,
                "RMSE":         rmse,
                "MAE":          mae
            })

# 6d. Summarize all runs in a DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
df_results.to_csv("dataframe.csv", index=False)
print(df_results)

print(f"Features: {X.shape[1]}")
import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Carga del dataset
df = pd.read_csv('ratings2comoML.csv')

# Pre-cálculo de soportes
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u) / total_users for m, u in users_by_movie.items()}

# Soporte mínimo y generación de pares y tríos frecuentes
min_support = 0.2
frequent_pairs = {
    frozenset({m1, m2}): len(users_by_movie[m1] & users_by_movie[m2]) / total_users
    for m1, m2 in combinations(users_by_movie.keys(), 2)
    if len(users_by_movie[m1] & users_by_movie[m2]) / total_users >= min_support
}
frequent_triples = {
    frozenset({m1, m2, m3}):
    len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users
    for m1, m2, m3 in combinations(users_by_movie.keys(), 3)
    if len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users >= min_support
}

# Función de extracción de 9 features Apriori
def apriori_features(row):
    user = row['userId']
    target = row['movieId']
    rated = set(df[df['userId'] == user]['movieId']) - {target}
    
    pair_count = pair_sum = 0
    confidences = []
    lifts = []
    triple_count = triple_sum = 0
    
    sup_target = movie_support.get(target, 0.0)
    
    for other in rated:
        pair = frozenset({target, other})
        if pair in frequent_pairs:
            sup = frequent_pairs[pair]
            pair_count += 1
            pair_sum += sup
            if sup_target > 0:
                confidences.append(sup / sup_target)
            o_sup = movie_support.get(other, 0)
            if sup_target > 0 and o_sup > 0:
                lifts.append(sup / (sup_target * o_sup))
    
    for combo in combinations(rated, 2):
        triple = frozenset({target, *combo})
        if triple in frequent_triples:
            triple_count += 1
            triple_sum += frequent_triples[triple]
    
    return pd.Series({
        'sup_target': sup_target,
        'cnt_rated': len(rated),
        'freq_pair_count': pair_count,
        'freq_pair_support_sum': pair_sum,
        'max_pair_confidence': max(confidences) if confidences else 0.0,
        'avg_pair_lift': sum(lifts)/len(lifts) if lifts else 0.0,
        'freq_triple_count': triple_count,
        'freq_triple_support_sum': triple_sum,
        'avg_triple_lift': sum(lifts)/len(lifts) if lifts else 0.0
    })

# Generar features Apriori
apriori_feats = df.apply(apriori_features, axis=1)
df_feat = pd.concat([df, apriori_feats], axis=1)

# Preparar X, y
df_final = df_feat.drop(['userId','movieId','timestamp'], axis=1)
X = df_final.drop('rating', axis=1)
y = df_final['rating']

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar XGBRegressor
xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                   objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)

# Predecir y evaluar
y_pred = xgb.predict(X_test)
rmse_xgb = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE XGBoost con 9 features Apriori: {rmse_xgb:.4f}")
print("Features:", len(X.columns))

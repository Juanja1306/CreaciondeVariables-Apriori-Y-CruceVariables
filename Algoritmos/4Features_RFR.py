import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# 1. Carga del dataset
df = pd.read_csv('ratings2comoML.csv')

# 2. Usuarios por película
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
total_users = df['userId'].nunique()
min_support = 0.2  # umbral de soporte mínimo

# 3. Generación de ítems frecuentes de tamaño 2 y 3
frequent_pairs = {}
frequent_triples = {}

# Ítems frecuentes de tamaño 2
for m1, m2 in combinations(users_by_movie.keys(), 2):
    inter = users_by_movie[m1] & users_by_movie[m2]
    support = len(inter) / total_users
    if support >= min_support:
        frequent_pairs[frozenset({m1, m2})] = support

# Ítems frecuentes de tamaño 3
for m1, m2, m3 in combinations(users_by_movie.keys(), 3):
    inter = users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]
    support = len(inter) / total_users
    if support >= min_support:
        frequent_triples[frozenset({m1, m2, m3})] = support

# 4. Función para extraer características basadas en Apriori
def apriori_features(row):
    user = row['userId']
    target = row['movieId']
    rated = set(df[df['userId'] == user]['movieId']) - {target}

    pair_count = 0
    pair_sum = 0.0
    triple_count = 0
    triple_sum = 0.0

    # Pares frecuentes
    for other in rated:
        pair = frozenset({target, other})
        if pair in frequent_pairs:
            pair_count += 1
            pair_sum += frequent_pairs[pair]

    # Tríos frecuentes
    for combo in combinations(rated, 2):
        triple = frozenset((target,) + combo)
        if triple in frequent_triples:
            triple_count += 1
            triple_sum += frequent_triples[triple]

    return pd.Series({
        'freq_pair_count': pair_count,
        'freq_pair_support_sum': pair_sum,
        'freq_triple_count': triple_count,
        'freq_triple_support_sum': triple_sum
    })

# 5. Aplicar características y preparar dataset final
apriori_feats = df.apply(apriori_features, axis=1)
df_feat = pd.concat([df, apriori_feats], axis=1)
df_final = df_feat.drop(['userId', 'movieId', 'timestamp'], axis=1)

X = df_final.drop('rating', axis=1)
y = df_final['rating']

# 6. Entrenar modelo y calcular RMSE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE: {rmse:.4f}")
print(f"Features: {X.shape[1]}")
import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 1. Carga del dataset
df = pd.read_csv('ratings2comoML.csv')

# 2. Pre-cálculo de soportes y sets de usuarios por película
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u) / total_users for m, u in users_by_movie.items()}

# 3. Ítems frecuentes de tamaño 2 y 3 (soporte ≥ 0.2)
min_support = 0.2
frequent_pairs = {
    frozenset([m1, m2]): len(users_by_movie[m1] & users_by_movie[m2]) / total_users
    for m1, m2 in combinations(users_by_movie, 2)
    if len(users_by_movie[m1] & users_by_movie[m2]) / total_users >= min_support
}
frequent_triples = {
    frozenset([m1, m2, m3]): len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users
    for m1, m2, m3 in combinations(users_by_movie, 3)
    if len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users >= min_support
}

# 4. Definición de la función de extracción de features Apriori
def apriori_features_ultimate(row):
    user = row['userId']
    target = row['movieId']
    rated = set(df[df['userId'] == user]['movieId']) - {target}
    
    sup_target = movie_support.get(target, 0.0)
    rated_count = len(rated)
    
    pair_count = pair_sum = 0
    triple_count = triple_sum = 0
    
    for other in rated:
        pair = frozenset([target, other])
        if pair in frequent_pairs:
            sup = frequent_pairs[pair]
            pair_count += 1
            pair_sum += sup
    
    for combo in combinations(rated, 2):
        tri = frozenset([target, *combo])
        if tri in frequent_triples:
            sup3 = frequent_triples[tri]
            triple_count += 1
            triple_sum += sup3
    
    return pd.Series({
        'sup_target': sup_target,
        'cnt_rated': rated_count,
        'freq_pair_count': pair_count,
        'freq_pair_support_sum': pair_sum,
        'freq_triple_count': triple_count,
        'freq_triple_support_sum': triple_sum
    })

# 5. Aplicar extracción de features Apriori
apr_feats = df.apply(apriori_features_ultimate, axis=1)
df_ml = pd.concat([df, apr_feats], axis=1)

# 6. Preparar X e y
X = df_ml.drop(['userId', 'movieId', 'timestamp', 'rating'], axis=1)
y = df_ml['rating']

# 7. División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Pipeline: escalado + KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(
        n_neighbors=10,
        weights='distance',
        n_jobs=-1
    ))
])

# 9. Entrenamiento y predicción
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 10. Evaluación RMSE
rmse_knn = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE KNN: {rmse_knn:.4f}")
print(f"Features: {len(X.columns)}")
import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 1. Carga de dataset de ratings
df = pd.read_csv('ratings2comoML.csv')

# 2. Carga de movies.csv con géneros
df_movies = pd.read_csv('movies.csv', usecols=['movieId','genres'], engine='python')
movie_genres = df_movies.set_index('movieId')['genres'].str.split('|').to_dict()

# 3. Pre-cálculo de soporte global y sets de usuarios por película
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u) / total_users for m, u in users_by_movie.items()}

# 4. Generar ítems frecuentes de tamaño 2 y 3 (soporte ≥ 0.2)
min_support = 0.2
frequent_pairs = {
    frozenset([m1, m2]): len(users_by_movie[m1] & users_by_movie[m2]) / total_users
    for m1, m2 in combinations(users_by_movie, 2)
    if len(users_by_movie[m1] & users_by_movie[m2]) / total_users >= min_support
}
frequent_triples = {
    frozenset([m1, m2, m3]):
    len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users
    for m1, m2, m3 in combinations(users_by_movie, 3)
    if len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users >= min_support
}

# 5. Función para extraer features Apriori + géneros
def extract_features(row):
    user, target = row['userId'], row['movieId']
    rated = set(df[df['userId'] == user]['movieId']) - {target}
    
    sup_target = movie_support.get(target, 0.0)
    cnt_rated = len(rated)
    pair_count = pair_sum = 0
    triple_count = triple_sum = 0
    
    # Pares frecuentes
    for other in rated:
        p = frozenset([target, other])
        if p in frequent_pairs:
            s = frequent_pairs[p]
            pair_count += 1
            pair_sum += s
    
    # Tríos frecuentes
    for combo in combinations(rated, 2):
        t = frozenset([target, *combo])
        if t in frequent_triples:
            s3 = frequent_triples[t]
            triple_count += 1
            triple_sum += s3
    
    feats = {
        'sup_target': sup_target,
        'cnt_rated': cnt_rated,
        'freq_pair_count': pair_count,
        'freq_pair_support_sum': pair_sum,
        'freq_triple_count': triple_count,
        'freq_triple_support_sum': triple_sum
    }
    
    # Features de géneros
    genres = movie_genres.get(target, [])
    feats['num_genres'] = len(genres)
    feats['is_genre_Comedy'] = int('Comedy' in genres)
    feats['is_genre_Drama'] = int('Drama' in genres)
    feats['is_genre_Action'] = int('Action' in genres)
    
    return pd.Series(feats)

# 6. Aplicar extracción y preparar X, y
features = df.apply(extract_features, axis=1)
df_ml = pd.concat([df, features], axis=1)
X = df_ml.drop(['userId', 'movieId', 'timestamp', 'rating'], axis=1)
y = df_ml['rating']

# 7. División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Pipeline KNN: escalado + KNeighborsRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=1))
])

# 9. Entrenar y predecir
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 10. Evaluación RMSE
rmse_knn = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE KNN (Apriori+Géneros, 10 vars): {rmse_knn:.4f}")
print("Features:", len(X.columns))
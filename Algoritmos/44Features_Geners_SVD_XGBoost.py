import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Carga de datos
df_ratings = pd.read_csv('ratings2comoML.csv')
df_movies  = pd.read_csv('movies.csv')   # columnas: movieId, title, genres

# 2. One-hot encoding de géneros
df_genres = df_movies['genres'].str.get_dummies(sep='|')
df_genres['movieId'] = df_movies['movieId']

# 3. Pre-cálculo de soportes para Apriori
total_users = df_ratings['userId'].nunique()
users_by_movie = df_ratings.groupby('movieId')['userId'].apply(set).to_dict()
movie_support   = {m: len(u)/total_users for m,u in users_by_movie.items()}
min_support     = 0.2

frequent_pairs = {
    frozenset([m1,m2]): len(users_by_movie[m1]&users_by_movie[m2]) / total_users
    for m1,m2 in combinations(users_by_movie,2)
    if len(users_by_movie[m1]&users_by_movie[m2]) / total_users >= min_support
}
frequent_triples = {
    frozenset([m1,m2,m3]):
      len(users_by_movie[m1]&users_by_movie[m2]&users_by_movie[m3]) / total_users
    for m1,m2,m3 in combinations(users_by_movie,3)
    if len(users_by_movie[m1]&users_by_movie[m2]&users_by_movie[m3]) / total_users >= min_support
}

# 4. Función para extraer features Apriori (pares + tríos)
def apriori_features(row):
    u, target = row['userId'], row['movieId']
    seen = set(df_ratings[df_ratings['userId']==u]['movieId']) - {target}
    pc = ps = tc = ts = 0
    sup_t = movie_support.get(target, 0.0)
    for other in seen:
        p = frozenset([target, other])
        if p in frequent_pairs:
            pc += 1
            ps += frequent_pairs[p]
    for a,b in combinations(seen,2):
        t = frozenset([target,a,b])
        if t in frequent_triples:
            tc += 1
            ts += frequent_triples[t]
    return pd.Series({
        'sup_target': sup_t,
        'cnt_rated': len(seen),
        'freq_pair_count': pc,
        'freq_pair_support_sum': ps,
        'freq_triple_count': tc,
        'freq_triple_support_sum': ts
    })

# 5. Aplicar Apriori y merge con géneros
apr_feats = df_ratings.apply(apriori_features, axis=1)
df = pd.concat([df_ratings, apr_feats], axis=1)
df = df.merge(df_genres, on='movieId', how='left')

# 6. Construir matriz usuario×película y extraer factores latentes
ui = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
k = min(20, ui.shape[1]-1, ui.shape[0]-1)
svd = TruncatedSVD(n_components=k, random_state=42)
U = svd.fit_transform(ui)             # (n_users, k)
V = svd.components_.T                # (n_movies, k)

df_U = pd.DataFrame(U, index=ui.index, columns=[f'u_lat_{i}' for i in range(k)])
df_V = pd.DataFrame(V, index=ui.columns, columns=[f'm_lat_{i}' for i in range(k)])

df = df.merge(df_U, left_on='userId',  right_index=True, how='left')
df = df.merge(df_V, left_on='movieId', right_index=True, how='left')

# 7. Preparar X e y
drop_cols = ['userId','movieId','timestamp','rating','title']
X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
y = df['rating']

# 8. División y entrenamiento con XGBoost
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
model.fit(X_tr, y_tr)

# 9. Predicción y cálculo de RMSE
y_pred = model.predict(X_te)
rmse = sqrt(mean_squared_error(y_te, y_pred))
print(f"RMSE Apriori + Géneros + SVD + XGBoost: {rmse:.4f}")
print(f"Features: {X_tr.shape[1]}")
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 1. Carga de datasets con bajo uso de memoria
df_ratings = pd.read_csv('ratings2comoML.csv')
df_movies = pd.read_csv('movies.csv', usecols=['movieId','genres'])

# 2. Preprocesar géneros manualmente
movie_genres = df_movies.set_index('movieId')['genres']\
                .apply(lambda s: s.split('|') if isinstance(s,str) else []).to_dict()
all_genres = sorted({g for genres in movie_genres.values() for g in genres})

# 3. Apriori: soporte global y frecuentes
total_users = df_ratings['userId'].nunique()
users_by_movie = df_ratings.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u)/total_users for m,u in users_by_movie.items()}
min_support = 0.2

# Pares/tríos frecuentes
frequent_pairs = {
    frozenset([m1,m2]): len(users_by_movie[m1]&users_by_movie[m2])/total_users
    for m1,m2 in combinations(users_by_movie,2)
    if len(users_by_movie[m1]&users_by_movie[m2])/total_users >= min_support
}
frequent_triples = {
    frozenset([m1,m2,m3]):
    len(users_by_movie[m1]&users_by_movie[m2]&users_by_movie[m3])/total_users
    for m1,m2,m3 in combinations(users_by_movie,3)
    if len(users_by_movie[m1]&users_by_movie[m2]&users_by_movie[m3])/total_users >= min_support
}

# 4. Función de features Apriori + géneros
def apriori_feats(row):
    user, target = row['userId'], row['movieId']
    rated = set(df_ratings[df_ratings['userId']==user]['movieId']) - {target}
    pc = ps = 0
    tc = ts = 0
    sup_target = movie_support.get(target,0.0)
    for other in rated:
        p = frozenset([target, other])
        if p in frequent_pairs:
            sup = frequent_pairs[p]
            pc += 1; ps += sup
    for combo in combinations(rated,2):
        t = frozenset([target,*combo])
        if t in frequent_triples:
            tc += 1; ts += frequent_triples[t]
    feats = {
        'sup_target': sup_target,
        'cnt_rated': len(rated),
        'freq_pair_count': pc,
        'freq_pair_support_sum': ps,
        'freq_triple_count': tc,
        'freq_triple_support_sum': ts
    }
    # añadir géneros one-hot
    genres = movie_genres.get(target, [])
    for g in all_genres:
        feats[f'genre_{g}'] = int(g in genres)
    return pd.Series(feats)

# 5. Generar features y merge
apriori_df = df_ratings.apply(apriori_feats, axis=1)
df_ml = pd.concat([df_ratings, apriori_df], axis=1)

# 6. Preparar X,e y
drop_cols = ['userId','movieId','timestamp']
X = df_ml.drop(drop_cols + ['rating'], axis=1)
y = df_ml['rating']

# 7. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Entrenar XGBoost
xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   objective='reg:squarederror', tree_method='hist',
                   random_state=42, n_jobs=4)
xgb.fit(X_train, y_train)

# 9. Evaluación
y_pred = xgb.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE XGBoost (Apriori + Géneros): {rmse:.4f}")
print(f"Features: {X.shape[1]}")
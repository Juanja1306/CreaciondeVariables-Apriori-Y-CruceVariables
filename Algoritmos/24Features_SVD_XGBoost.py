import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 1. Carga del dataset
df = pd.read_csv('ratings2comoML.csv')

# 2. Usuario-por-película y soporte global
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u) / total_users for m, u in users_by_movie.items()}

# 3. Generar ítems frecuentes de tamaño 2 y 3 (soporte ≥ 0.2)
min_support = 0.2
frequent_pairs = {
    frozenset([m1, m2]): len(users_by_movie[m1] & users_by_movie[m2]) / total_users
    for m1, m2 in combinations(users_by_movie.keys(), 2)
    if len(users_by_movie[m1] & users_by_movie[m2]) / total_users >= min_support
}
frequent_triples = {
    frozenset([m1, m2, m3]): len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users
    for m1, m2, m3 in combinations(users_by_movie.keys(), 3)
    if len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users >= min_support
}

# 4. Función Apriori para pares y tríos
def apriori_features(row):
    user = row['userId']
    target = row['movieId']
    rated = set(df[df['userId'] == user]['movieId']) - {target}
    pc = ps = tc = ts = 0
    for other in rated:
        pair = frozenset([target, other])
        if pair in frequent_pairs:
            pc += 1
            ps += frequent_pairs[pair]
    for combo in combinations(rated, 2):
        tri = frozenset([target] + list(combo))
        if tri in frequent_triples:
            tc += 1
            ts += frequent_triples[tri]
    return pd.Series({
        'sup_target': movie_support.get(target, 0.0),
        'cnt_rated': len(rated),
        'freq_pair_count': pc,
        'freq_pair_support_sum': ps,
        'freq_triple_count': tc,
        'freq_triple_support_sum': ts
    })

# 5. Aplicar Apriori
apriori_feats = df.apply(apriori_features, axis=1)
df_feat = pd.concat([df, apriori_feats], axis=1)

# 6. Matriz usuario×película y SVD
user_item = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
n_comp = min(20, user_item.shape[1] - 1, user_item.shape[0] - 1)
svd = TruncatedSVD(n_components=n_comp, random_state=42)
user_latent = svd.fit_transform(user_item)
movie_latent = svd.components_.T

user_latent_df = pd.DataFrame(user_latent, index=user_item.index,
                              columns=[f'u_lat_{i}' for i in range(n_comp)])
movie_latent_df = pd.DataFrame(movie_latent, index=user_item.columns,
                               columns=[f'm_lat_{i}' for i in range(n_comp)])

df_feat = df_feat.merge(user_latent_df, left_on='userId', right_index=True)
df_feat = df_feat.merge(movie_latent_df, left_on='movieId', right_index=True)

# 7. Preparar X, y
df_final = df_feat.drop(['userId', 'movieId', 'timestamp'], axis=1)
X = df_final.drop('rating', axis=1)
y = df_final['rating']

# 8. División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Entrenar XGBoost
xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                   objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
rmse_xgb = sqrt(mean_squared_error(y_test, pred_xgb))

print(f"RMSE XGBoost: {rmse_xgb:.4f}")
print(f"Features: {X_train.shape[1]}")
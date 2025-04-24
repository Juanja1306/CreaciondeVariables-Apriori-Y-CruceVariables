import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Leer ratings y movies (con géneros)
df_ratings = pd.read_csv('ratings2comoML.csv')
df_movies  = pd.read_csv(r"C:\Users\juanj\Desktop\ml-32m\movies.csv")  # movieId,title,genres

# 2. Preprocesar géneros: one-hot encoding
#    Cada película puede tener múltiples géneros separados por '|'
genres_expanded = df_movies['genres'].str.get_dummies(sep='|')
df_genres = pd.concat([df_movies[['movieId']], genres_expanded], axis=1)

# 3. Calcular Apriori features (pares + tríos) — igual que tu código
total_users = df_ratings['userId'].nunique()
users_by_movie = df_ratings.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u)/total_users for m,u in users_by_movie.items()}
min_support = 0.2

# ítems frecuentes
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

def apriori_feats(row):
    user,target = row['userId'], row['movieId']
    rated = set(df_ratings[df_ratings['userId']==user]['movieId']) - {target}
    pc = ps = tc = ts = 0
    for other in rated:
        p = frozenset([target,other])
        if p in frequent_pairs:
            pc += 1
            ps += frequent_pairs[p]
    for combo in combinations(rated,2):
        t = frozenset([target,*combo])
        if t in frequent_triples:
            tc += 1
            ts += frequent_triples[t]
    return pd.Series({
        'sup_target': movie_support.get(target,0),
        'cnt_rated': len(rated),
        'freq_pair_count': pc,
        'freq_pair_support_sum': ps,
        'freq_triple_count': tc,
        'freq_triple_support_sum': ts
    })

# 4. Construir DF con features Apriori
apriori_features = df_ratings.apply(apriori_feats, axis=1)
df = pd.concat([df_ratings, apriori_features], axis=1)

# 5. Merge con géneros
df = df.merge(df_genres, on='movieId')

# 6. Preparar X e y
X = df.drop(['rating', 'userId', 'movieId', 'timestamp', 'title'] if 'title' in df.columns else ['rating','userId','movieId','timestamp'], axis=1)
y = df['rating']

# 7. Train/Test split y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=549, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

# 8. Evaluación
rmse = sqrt(mean_squared_error(y_test, pred))
print(f"RMSE con géneros + Apriori: {rmse:.4f}")
print(f"Features: {X_train.shape[1]}")
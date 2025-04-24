import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1. Carga de datos
df_ratings = pd.read_csv('ratings2comoML.csv')
df_movies  = pd.read_csv('movies.csv', usecols=['movieId','genres'])
movie_genres = df_movies.set_index('movieId')['genres'].str.split('|').to_dict()

# 2. Pre-cálculo de soportes
total_users = df_ratings['userId'].nunique()
users_by_movie = df_ratings.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u)/total_users for m,u in users_by_movie.items()}

# 3. Ítems frecuentes (pares y tríos)
min_support = 0.2
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

# 4. Extracción de las 10 features
def extract_features(row):
    user, target = row['userId'], row['movieId']
    rated = set(df_ratings[df_ratings['userId']==user]['movieId']) - {target}
    sup_t = movie_support.get(target,0.0)
    cnt_r = len(rated)
    pc = ps = tc = ts = 0
    for other in rated:
        p = frozenset([target,other])
        if p in frequent_pairs:
            s = frequent_pairs[p]
            pc += 1; ps += s
    for combo in combinations(rated,2):
        t = frozenset([target,*combo])
        if t in frequent_triples:
            s3 = frequent_triples[t]
            tc += 1; ts += s3
    feats = {
        'sup_target': sup_t,
        'cnt_rated': cnt_r,
        'freq_pair_count': pc,
        'freq_pair_support_sum': ps,
        'freq_triple_count': tc,
        'freq_triple_support_sum': ts,
    }
    genres = movie_genres.get(target, [])
    feats['num_genres']   = len(genres)
    feats['is_Comedy']    = int('Comedy' in genres)
    feats['is_Drama']     = int('Drama' in genres)
    feats['is_Thriller']  = int('Thriller' in genres)
    return pd.Series(feats)

feature_df = df_ratings.apply(extract_features, axis=1)
X = feature_df.values
y = df_ratings['rating'].values

# 5. Escalado y reshape para RNN (timesteps=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 6. Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42
)

# 7. Definir y entrenar la RNN
n_features = X_seq.shape[2]
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(1, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Ajusta epochs/batch_size según tu máquina
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 8. Evaluación
y_pred = model.predict(X_test).flatten()
rmse_rnn = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE RNN: {rmse_rnn:.4f}")
print(f"Numero de features: {X.shape[1]}")
import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 1. Carga del dataset
df = pd.read_csv('ratings2comoML.csv')

# 2. Usuario-por-película y soportes
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support   = {m: len(u)/total_users for m,u in users_by_movie.items()}

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

# 4. Función de extracción de las 21 features Apriori avanzadas
def apriori_features_ultimate(row):
    user   = row['userId']
    target = row['movieId']
    rated  = set(df[df['userId']==user]['movieId']) - {target}

    sup_t = movie_support.get(target, 0.0)
    pc = ps = tc = ts = 0

    for other in rated:
        p = frozenset([target, other])
        if p in frequent_pairs:
            s = frequent_pairs[p]
            pc += 1; ps += s

    for combo in combinations(rated,2):
        t = frozenset([target,*combo])
        if t in frequent_triples:
            s3 = frequent_triples[t]
            tc += 1; ts += s3

    return pd.Series({
        'sup_target': sup_t,
        'cnt_rated': len(rated),
        'freq_pair_count': pc,
        'freq_pair_support_sum': ps,
        'freq_triple_count': tc,
        'freq_triple_support_sum': ts,
        'max_pair_support': max((frequent_pairs[frozenset([target,o])] 
                                 for o in rated if frozenset([target,o]) in frequent_pairs), default=0.0),
        'avg_pair_support': (ps/pc) if pc>0 else 0.0,
        'sum_pair_leverage': sum((frequent_pairs[frozenset([target,o])] -
                                  sup_t * movie_support[o])
                                 for o in rated if frozenset([target,o]) in frequent_pairs),
        'max_pair_confidence': max(((frequent_pairs[frozenset([target,o])] / sup_t)
                                    for o in rated if sup_t>0 and frozenset([target,o]) in frequent_pairs),
                                   default=0.0),
        'avg_pair_lift': sum((frequent_pairs[frozenset([target,o])] /
                              (sup_t*movie_support[o]))
                             for o in rated if sup_t>0 and movie_support[o]>0 and frozenset([target,o]) in frequent_pairs)
                          / max(pc,1),
        'freq_triple_support_sum': ts,
        'triple_coverage': tc / max((len(rated)*(len(rated)-1)/2),1)
    })

# 5. Generar dataset de features
ult_feats = df.apply(apriori_features_ultimate, axis=1)
df_ml = pd.concat([df, ult_feats], axis=1).drop(['userId','movieId','timestamp'], axis=1)

X = df_ml.drop('rating', axis=1)
y = df_ml['rating']

# 6. División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Pipeline: escalado + MLPRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp',    MLPRegressor(
                   hidden_layer_sizes=(100,),
                   activation='relu',
                   solver='adam',
                   max_iter=200,
                   random_state=42
               ))
])

# 8. Entrenamiento y evaluación
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
rmse_nn = sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE Neural Network: {rmse_nn:.4f}")
print(f"Features: {X.shape[1]}")
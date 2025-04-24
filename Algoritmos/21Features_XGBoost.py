import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 1. Carga del dataset de ratings
df = pd.read_csv('ratings2comoML.csv')

# 2. Soporte global y sets de usuarios por película
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u)/total_users for m,u in users_by_movie.items()}

# 3. Generar ítems frecuentes de tamaño 2 y 3
min_support = 0.2
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

# 4. Función Apriori features ultimate
def apriori_features_ultimate(row):
    user = row['userId']
    target = row['movieId']
    rated = set(df[df['userId']==user]['movieId']) - {target}
    
    sup_target = movie_support.get(target, 0.0)
    cnt_rated = len(rated)
    
    pair_supports = []
    pair_leverages = []
    confs = []
    lifts = []
    weighted_ratings = []
    triple_supports = []
    triple_leverages = []
    triple_lifts = []
    
    for other in rated:
        p = frozenset([target,other])
        if p in frequent_pairs:
            sup = frequent_pairs[p]
            pair_supports.append(sup)
            o_sup = movie_support.get(other, 0)
            pair_leverages.append(sup - sup_target*o_sup)
            if sup_target>0:
                confs.append(sup/sup_target)
            if sup_target>0 and o_sup>0:
                lifts.append(sup/(sup_target*o_sup))
            rating_other = df[(df['userId']==user)&(df['movieId']==other)]['rating'].iloc[0]
            weighted_ratings.append(sup*rating_other)
    for combo in combinations(rated,2):
        t = frozenset([target,*combo])
        if t in frequent_triples:
            sup3 = frequent_triples[t]
            triple_supports.append(sup3)
            o1,o2 = combo
            o1_sup = movie_support.get(o1,0)
            o2_sup = movie_support.get(o2,0)
            triple_leverages.append(sup3 - sup_target*o1_sup*o2_sup)
            if sup_target>0 and o1_sup>0 and o2_sup>0:
                triple_lifts.append(sup3/(sup_target*o1_sup*o2_sup))
    return pd.Series({
        'sup_target': sup_target,
        'cnt_rated': cnt_rated,
        'freq_pair_count': len(pair_supports),
        'freq_pair_support_sum': sum(pair_supports),
        'max_pair_support': max(pair_supports) if pair_supports else 0,
        'min_pair_support': min(pair_supports) if pair_supports else 0,
        'avg_pair_support': sum(pair_supports)/len(pair_supports) if pair_supports else 0,
        'sum_pair_leverage': sum(pair_leverages),
        'max_pair_leverage': max(pair_leverages) if pair_leverages else 0,
        'max_pair_confidence': max(confs) if confs else 0,
        'avg_pair_lift': sum(lifts)/len(lifts) if lifts else 0,
        'max_pair_lift': max(lifts) if lifts else 0,
        'weighted_avg_rating_pair': sum(weighted_ratings)/sum(pair_supports) if pair_supports else 0,
        'freq_triple_count': len(triple_supports),
        'freq_triple_support_sum': sum(triple_supports),
        'avg_triple_support': sum(triple_supports)/len(triple_supports) if triple_supports else 0,
        'max_triple_support': max(triple_supports) if triple_supports else 0,
        'sum_triple_leverage': sum(triple_leverages),
        'max_triple_lift': max(triple_lifts) if triple_lifts else 0,
        'avg_triple_lift': sum(triple_lifts)/len(triple_lifts) if triple_lifts else 0,
        'triple_coverage': len(triple_supports)/(cnt_rated*(cnt_rated-1)/2) if cnt_rated>1 else 0
    })

# 5. Aplicar y preparar dataset
feat_df = df.apply(apriori_features_ultimate, axis=1)
df_ml = pd.concat([df, feat_df], axis=1)
X = df_ml.drop(['userId','movieId','timestamp','rating'], axis=1)
y = df_ml['rating']

# 6. División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Entrenar XGBoost con parámetros ajustados
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    tree_method='hist',
    random_state=42,
    n_jobs=4
)
xgb.fit(X_train, y_train)

# 8. Evaluar RMSE
y_pred = xgb.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE XGBoost (Apriori ultimate): {rmse:.4f}")
print(f"Features: {X_train.shape[1]}")
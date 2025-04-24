from itertools import combinations
import joblib
import pandas as pd

# 1. Carga del modelo entrenado
rf = joblib.load('rf_model.pkl')

# 2. Carga del dataset original para reconstruir soportes y frecuentes
df = pd.read_csv(r'./CSVs/ratings2comoML.csv')

# 3. Cálculo de soportes globales y sets de usuarios por película
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u)/total_users for m, u in users_by_movie.items()}

# 4. Generar ítems frecuentes de tamaño 2 y 3 (soporte ≥ 0.2)
min_support = 0.2
frequent_pairs = {
    frozenset({m1, m2}): len(users_by_movie[m1] & users_by_movie[m2]) / total_users
    for m1, m2 in combinations(users_by_movie.keys(), 2)
    if len(users_by_movie[m1] & users_by_movie[m2]) / total_users >= min_support
}
frequent_triples = {
    frozenset({m1, m2, m3}):
    len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users
    for m1, m2, m3 in combinations(users_by_movie.keys(), 3)
    if len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users >= min_support
}

# 5. Función para extraer las mismas 21 features
def apriori_features_ultimate(row):
    user = row['userId']
    target = row['movieId']
    rated = set(df[df['userId'] == user]['movieId']) - {target}
    
    sup_target = movie_support.get(target, 0.0)
    cnt_rated = len(rated)
    
    pair_supports, pair_leverages, confs, lifts, weighted_ratings = [], [], [], [], []
    triple_supports, triple_leverages, triple_lifts = [], [], []
    
    for other in rated:
        p = frozenset({target, other})
        if p in frequent_pairs:
            sup = frequent_pairs[p]
            pair_supports.append(sup)
            o_sup = movie_support.get(other, 0)
            pair_leverages.append(sup - sup_target * o_sup)
            if sup_target > 0:
                confs.append(sup / sup_target)
            if sup_target > 0 and o_sup > 0:
                lifts.append(sup / (sup_target * o_sup))
            rating_other = df[(df['userId'] == user) & (df['movieId'] == other)]['rating'].iloc[0]
            weighted_ratings.append(sup * rating_other)
    
    for combo in combinations(rated, 2):
        t = frozenset({target, *combo})
        if t in frequent_triples:
            sup3 = frequent_triples[t]
            triple_supports.append(sup3)
            o1, o2 = combo
            o1_sup = movie_support.get(o1, 0)
            o2_sup = movie_support.get(o2, 0)
            triple_leverages.append(sup3 - sup_target * o1_sup * o2_sup)
            if sup_target > 0 and o1_sup > 0 and o2_sup > 0:
                triple_lifts.append(sup3 / (sup_target * o1_sup * o2_sup))
    
    return pd.Series({
        'sup_target': sup_target,
        'cnt_rated': cnt_rated,
        'freq_pair_count': len(pair_supports),
        'freq_pair_support_sum': sum(pair_supports),
        'max_pair_support': max(pair_supports) if pair_supports else 0.0,
        'min_pair_support': min(pair_supports) if pair_supports else 0.0,
        'avg_pair_support': sum(pair_supports)/len(pair_supports) if pair_supports else 0.0,
        'sum_pair_leverage': sum(pair_leverages),
        'max_pair_leverage': max(pair_leverages) if pair_leverages else 0.0,
        'max_pair_confidence': max(confs) if confs else 0.0,
        'avg_pair_lift': sum(lifts)/len(lifts) if lifts else 0.0,
        'max_pair_lift': max(lifts) if lifts else 0.0,
        'weighted_avg_rating_pair': sum(weighted_ratings)/sum(pair_supports) if pair_supports else 0.0,
        'freq_triple_count': len(triple_supports),
        'freq_triple_support_sum': sum(triple_supports),
        'avg_triple_support': sum(triple_supports)/len(triple_supports) if triple_supports else 0.0,
        'max_triple_support': max(triple_supports) if triple_supports else 0.0,
        'sum_triple_leverage': sum(triple_leverages),
        'max_triple_lift': max(triple_lifts) if triple_lifts else 0.0,
        'avg_triple_lift': sum(triple_lifts)/len(triple_lifts) if triple_lifts else 0.0,
        'triple_coverage': len(triple_supports)/(cnt_rated*(cnt_rated-1)/2) if cnt_rated > 1 else 0.0
    })

# 6. Función para predecir
def predict_rating(user_id, movie_id):
    row = pd.Series({'userId': user_id, 'movieId': movie_id})
    feats = apriori_features_ultimate(row)
    X_new = pd.DataFrame([feats])
    return rf.predict(X_new)[0]

# 7. Uso interactivo
if __name__ == "__main__":
    uid = int(input("User ID: "))
    mid = int(input("Movie ID: "))
    pred = predict_rating(uid, mid)
    print(f"Predicción de rating para user {uid}, movie {mid}: {pred:.3f}")
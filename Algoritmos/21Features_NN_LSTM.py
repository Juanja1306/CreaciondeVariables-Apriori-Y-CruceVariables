import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Carga del dataset
df = pd.read_csv('ratings2comoML.csv')

# 2. Usuario-por-película y soportes
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
movie_support = {m: len(u) / total_users for m, u in users_by_movie.items()}

# 3. Generar ítems frecuentes de tamaño 2 y 3 (soporte ≥ 0.2)
min_support = 0.2
frequent_pairs = {frozenset({m1, m2}): len(users_by_movie[m1] & users_by_movie[m2]) / total_users
                  for m1, m2 in combinations(users_by_movie.keys(), 2)
                  if len(users_by_movie[m1] & users_by_movie[m2]) / total_users >= min_support}
frequent_triples = {frozenset({m1, m2, m3}):
                    len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users
                    for m1, m2, m3 in combinations(users_by_movie.keys(), 3)
                    if len(users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]) / total_users >= min_support}

# 4. Extracción de features avanzadas
def apriori_features_ultimate(row):
    user = row['userId']
    target = row['movieId']
    rated = set(df[df['userId'] == user]['movieId']) - {target}
    
    # Unarios
    sup_target = movie_support.get(target, 0.0)
    cnt_rated = len(rated)
    
    # Inicializar
    pair_supports = []
    pair_leverages = []
    confs = []
    lifts = []
    weighted_ratings = []
    
    triple_supports = []
    triple_leverages = []
    triple_lifts = []
    
    for other in rated:
        pair = frozenset({target, other})
        if pair in frequent_pairs:
            sup = frequent_pairs[pair]
            pair_supports.append(sup)
            t_sup = sup_target
            o_sup = movie_support.get(other, 0)
            pair_leverages.append(sup - t_sup * o_sup)
            if t_sup > 0:
                confs.append(sup / t_sup)
            if t_sup > 0 and o_sup > 0:
                lifts.append(sup / (t_sup * o_sup))
            # Weighted by support
            rating_other = df[(df['userId'] == user) & (df['movieId'] == other)]['rating'].iloc[0]
            weighted_ratings.append(sup * rating_other)
    
    for combo in combinations(rated, 2):
        triple = frozenset((target,) + combo)
        if triple in frequent_triples:
            sup3 = frequent_triples[triple]
            triple_supports.append(sup3)
            t_sup = sup_target
            o1_sup = movie_support.get(combo[0], 0)
            o2_sup = movie_support.get(combo[1], 0)
            triple_leverages.append(sup3 - t_sup * o1_sup * o2_sup)
            if t_sup > 0 and o1_sup > 0 and o2_sup > 0:
                triple_lifts.append(sup3 / (t_sup * o1_sup * o2_sup))
    
    # Features cálculo
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
        'triple_coverage': len(triple_supports)/ (cnt_rated*(cnt_rated-1)/2) if cnt_rated > 1 else 0.0
    })

# 5. Generar dataset final
ultimate_feats = df.apply(apriori_features_ultimate, axis=1)
df_ult = pd.concat([df, ultimate_feats], axis=1).drop(['userId','movieId','timestamp'], axis=1)
X = df_ult.drop('rating', axis=1)
y = df_ult['rating']

# 6. Dividir y entrenar modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 1) Convertir DataFrames a numpy float32
X_train_np = X_train.to_numpy(dtype=np.float32)
y_train_np = y_train.to_numpy(dtype=np.float32)
X_test_np  = X_test.to_numpy(dtype=np.float32)
y_test_np  = y_test.to_numpy(dtype=np.float32)

# 2) Crear tensores y añadir dimensión de “feature”
#    De (N, seq_len) a (N, seq_len, 1)
X_train_t = torch.from_numpy(X_train_np).unsqueeze(2)
y_train_t = torch.from_numpy(y_train_np)
X_test_t  = torch.from_numpy(X_test_np).unsqueeze(2)
y_test_t  = torch.from_numpy(y_test_np)

# 3) DataLoader
batch_size = 512
train_ds   = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 4) Definir el modelo RNN (LSTM + lineal)
class RNNRegressor(nn.Module):
    def __init__(self, seq_len, hidden_size=64, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)               # (batch, seq_len, hidden_size)
        last = out[:, -1, :]                # tomar salida del último paso
        return self.fc(last).squeeze(1)     # (batch,)

# 5) Preparar dispositivo y modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = X_train_t.shape[1]
model   = RNNRegressor(seq_len=seq_len, hidden_size=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 6) Entrenamiento
n_epochs = 100
for epoch in range(1, n_epochs+1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    mse_train = running_loss / len(train_loader.dataset)
    # print(f"Epoch {epoch:02d} — MSE train: {mse_train:.4f}")

# 7) Evaluación en test
model.eval()
with torch.no_grad():
    preds_test = model(X_test_t.to(device)).cpu().numpy()
rmse_rnn = sqrt(mean_squared_error(y_test_np, preds_test))
print(f"\nRMSE RNN (LSTM): {rmse_rnn:.4f}")
print(f"Features: {X.shape[1]}")
import pandas as pd
from itertools import combinations
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 1. Carga y features base (idéntico a tu código)
df = pd.read_csv('ratings2comoML.csv')
df['user_avg_rating'] = df.groupby('userId')['rating'].transform('mean')
df['movie_avg_rating'] = df.groupby('movieId')['rating'].transform('mean')
df['rating_diff'] = df['user_avg_rating'] - df['movie_avg_rating']

# 2. Apriori básico (idéntico)
total_users = df['userId'].nunique()
users_by_movie = df.groupby('movieId')['userId'].apply(set).to_dict()
min_support = 0.2

frequent_pairs = {}
for m1, m2 in combinations(users_by_movie, 2):
    inter = users_by_movie[m1] & users_by_movie[m2]
    support = len(inter) / total_users
    if support >= min_support:
        frequent_pairs[frozenset({m1, m2})] = support

frequent_triples = {}
for m1, m2, m3 in combinations(users_by_movie, 3):
    inter = users_by_movie[m1] & users_by_movie[m2] & users_by_movie[m3]
    support = len(inter) / total_users
    if support >= min_support:
        frequent_triples[frozenset({m1, m2, m3})] = support

def apriori_basic(row):
    user, target = row['userId'], row['movieId']
    rated = set(df[df['userId']==user]['movieId']) - {target}
    pc = ps = tc = ts = 0
    for other in rated:
        pair = frozenset({target, other})
        if pair in frequent_pairs:
            pc += 1
            ps += frequent_pairs[pair]
    for combo in combinations(rated, 2):
        tri = frozenset((target,)+combo)
        if tri in frequent_triples:
            tc += 1
            ts += frequent_triples[tri]
    return pd.Series({
        'freq_pair_count': pc,
        'freq_pair_support_sum': ps,
        'freq_triple_count': tc,
        'freq_triple_support_sum': ts
    })

apriori_feats = df.apply(apriori_basic, axis=1)
df_model = pd.concat([df, apriori_feats], axis=1)

# 3. Preparamos tensores para PyTorch
X = df_model[['user_avg_rating','movie_avg_rating','rating_diff',
              'freq_pair_count','freq_pair_support_sum',
              'freq_triple_count','freq_triple_support_sum']].values
y = df_model['rating'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convertimos a tensores
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=1024, shuffle=True
)

# 4. Definimos el modelo FM
class FactorizationMachine(nn.Module):
    def __init__(self, n_features, k):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        # Factores latentes V: [n_features × k]
        self.v = nn.Parameter(torch.randn(n_features, k) * 0.01)

    def forward(self, x):
        # parte lineal
        lin = self.linear(x)  # [batch,1]
        # interacción de segundo orden:
        # ( (xV)^2 - (x^2 V^2) ).sum(dim=1) * 0.5
        xv = x @ self.v              # [batch, k]
        xv2 = (x**2) @ (self.v**2)   # [batch, k]
        interactions = 0.5 * torch.sum(xv**2 - xv2, dim=1, keepdim=True)
        return lin + interactions

# 5. Entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FactorizationMachine(n_features=X_train.shape[1], k=10).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(30):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    #print(f"Epoch {epoch+1:02d}  MSE train: {total_loss/len(train_loader.dataset):.4f}")

# 6. Evaluación
model.eval()
with torch.no_grad():
    preds_test = model(X_test_t.to(device)).squeeze().cpu().numpy()
rmse = sqrt(mean_squared_error(y_test, preds_test))
print(f"\nRMSE con FM PyTorch: {rmse:.4f}")
print(f"Features: {X.shape[1]}")
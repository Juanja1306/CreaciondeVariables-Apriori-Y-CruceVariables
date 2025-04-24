import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Carga y limpieza igual que antes  
df = pd.read_csv('ratings2comoML.csv').drop('timestamp', axis=1)

# 1) Calcula la media global y su RMSE (baseline)
global_mean = df['rating'].mean()
y_true = df['rating']
y_pred_baseline = np.full_like(y_true, global_mean, dtype=float)
baseline_rmse = np.sqrt(mean_squared_error(y_true, y_pred_baseline))

print(f'RMSE baseline (media global={global_mean:.4f}): {baseline_rmse:.4f}')

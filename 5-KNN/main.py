# main_vectorized.py
"""
Teste do KNN vetorizado (implementado do zero) comparado com scikit-learn.
Inclui métricas reais e uso de bases reais.
"""

import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Importar funções vetorizadas
from knn_algorithm import knn_classifier_vectorized, knn_regressor_vectorized


print("=== CLASSIFICAÇÃO: Iris Dataset ===")
iris = load_iris()
X, y = iris.data, iris.target

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_class = 5

y_pred_custom = knn_classifier_vectorized(X_train_scaled, y_train, X_test_scaled, k_class)

# --- KNN do sklearn ---
knn_sklearn = KNeighborsClassifier(n_neighbors=k_class, metric='euclidean')
knn_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test_scaled)

# --- Métricas ---
print("\nResultados KNN implementado vetorizado:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_custom):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred_custom, average='macro'):.4f}")

print("\nResultados KNN scikit-learn:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_sklearn):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred_sklearn, average='macro'):.4f}")

# ----------------------------------------------------------
# 2. REGRESSÃO: California Housing
# ----------------------------------------------------------
print("\n=== REGRESSÃO: California Housing Dataset ===")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_reg = 5

# --- KNN implementado vetorizado ---
y_pred_custom_reg = knn_regressor_vectorized(X_train_scaled, y_train, X_test_scaled, k_reg)

# --- KNN do sklearn ---
knn_sklearn_reg = KNeighborsRegressor(n_neighbors=k_reg, metric='euclidean')
knn_sklearn_reg.fit(X_train_scaled, y_train)
y_pred_sklearn_reg = knn_sklearn_reg.predict(X_test_scaled)

# --- Métricas ---
mse_custom = mean_squared_error(y_test, y_pred_custom_reg)
rmse_custom = np.sqrt(mse_custom)
r2_custom = r2_score(y_test, y_pred_custom_reg)

mse_sklearn = mean_squared_error(y_test, y_pred_sklearn_reg)
rmse_sklearn = np.sqrt(mse_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn_reg)

print("\nResultados KNN implementado vetorizado:")
print(f"RMSE: {rmse_custom:.4f}, R²: {r2_custom:.4f}")

print("\nResultados KNN scikit-learn:")
print(f"RMSE: {rmse_sklearn:.4f}, R²: {r2_sklearn:.4f}")

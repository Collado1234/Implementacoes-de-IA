import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris, load_diabetes 
from sklearn.tree import DecisionTreeClassifier as SklearnDTC 
from sklearn.tree import DecisionTreeRegressor as SklearnDTR 
from sklearn.metrics import confusion_matrix, classification_report, r2_score 


# Importando suas implementações 
from decision_tree_classifier import DecisionTreeClassifier
from decision_tree_regressor import DecisionTreeRegressor, mean_squared_error_manual


# =============================================================
# MÓDULO DE TESTE 1: CLASSIFICAÇÃO (IRIS Dataset) - DETALHADO 📊
# =============================================================
def test_classification_detailed():
    print("="*15 + " TESTE CLASSIFICAÇÃO (IRIS Dataset) DETALHADO " + "="*15)
    
    # 1. Carregar Dados e Dividir
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    MAX_DEPTH = 3 
    
    # --- 2. SEU MODELO (Manual) ---
    print("\n--- 🌳 Seu DecisionTreeClassifier ---")
    clf_manual = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='gini')
    clf_manual.fit(X_train, y_train)
    y_pred_manual = clf_manual.predict(X_test)
    
    print(f"Acurácia no Teste: {accuracy_score(y_test, y_pred_manual):.4f}")
    print("\nRelatório de Classificação (Seu Código):\n")
    print(classification_report(y_test, y_pred_manual, target_names=target_names, zero_division=0))
    print("\nMatriz de Confusão (Seu Código):\n", confusion_matrix(y_test, y_pred_manual))
    
    # --- 3. Modelo Sklearn ---
    print("\n--- 🚀 Sklearn DecisionTreeClassifier ---")
    clf_sklearn = SklearnDTC(max_depth=MAX_DEPTH, criterion='gini', random_state=42)
    clf_sklearn.fit(X_train, y_train)
    y_pred_sklearn = clf_sklearn.predict(X_test)
    
    print(f"Acurácia no Teste: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print("\nRelatório de Classificação (Sklearn):\n")
    print(classification_report(y_test, y_pred_sklearn, target_names=target_names, zero_division=0))
    print("\nMatriz de Confusão (Sklearn):\n", confusion_matrix(y_test, y_pred_sklearn))
    
    # --- 4. Comparação ---
    print("\n" + "-"*50)
    print("| Resumo (Acurácia) | Manual: {0:.4f} | Sklearn: {1:.4f} |".format(
        accuracy_score(y_test, y_pred_manual), 
        accuracy_score(y_test, y_pred_sklearn)
    ))
    print("-"*50)

# =============================================================
# MÓDULO DE TESTE 2: REGRESSÃO (DIABETES Dataset) - DETALHADO 📈
# =============================================================
def test_regression_detailed():
    print("\n\n" + "="*15 + " TESTE REGRESSÃO (DIABETES Dataset) DETALHADO " + "="*15)
    
    # 1. Carregar Dados e Dividir
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    MAX_DEPTH = 4 
    
    # --- 2. SEU MODELO (Manual) ---
    print("\n--- 🌳 Seu DecisionTreeRegressor ---")
    reg_manual = DecisionTreeRegressor(max_depth=MAX_DEPTH, criterion='mse')
    reg_manual.fit(X_train, y_train)
    
    y_pred_manual = reg_manual.predict(X_test)
    mse_manual = mean_squared_error_manual(y_test, y_pred_manual)
    r2_manual = r2_score(y_test, y_pred_manual)
    
    print(f"Seu MSE no Teste: {mse_manual:.2f}")
    print(f"Seu Coeficiente de Determinação (R²): {r2_manual:.4f}")
    
    # --- 3. Modelo Sklearn ---
    print("\n--- 🚀 Sklearn DecisionTreeRegressor ---")
    # CORREÇÃO APLICADA: Deve treinar com X_train e y_train
    reg_sklearn = SklearnDTR(max_depth=MAX_DEPTH, criterion='squared_error', random_state=42)
    reg_sklearn.fit(X_train, y_train) 
    
    y_pred_sklearn = reg_sklearn.predict(X_test)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    
    print(f"Sklearn MSE no Teste: {mse_sklearn:.2f}")
    print(f"Sklearn Coeficiente de Determinação (R²): {r2_sklearn:.4f}")

    # --- 4. Comparação ---
    print("\n" + "-"*50)
    print("| Comparação de Métricas de Regressão (Depth=4) |")
    print(f"| MSE Manual: {mse_manual:.2f} | MSE Sklearn: {mse_sklearn:.2f} |")
    print(f"| R² Manual: {r2_manual:.4f} | R² Sklearn: {r2_sklearn:.4f} |")
    print("-"*50)

# =============================================================
# EXECUÇÃO PRINCIPAL
# =============================================================
if __name__ == "__main__":
    test_classification_detailed()
    test_regression_detailed()
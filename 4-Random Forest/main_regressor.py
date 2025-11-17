# test_regressor.py

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor as SklearnRFR
# Assumindo que suas classes estão importáveis:
from decision_tree_regressor import DecisionTreeRegressor 
from random_forest_regressor import RandomForestRegressor 
# Se as classes estiverem no mesmo arquivo, remova estes imports.

# =============================================================
# FUNÇÃO PRINCIPAL DE TESTE DE REGRESSÃO
# =============================================================

def test_random_forest_regressor(ManualDTR, ManualRFR):
    print("="*10 + " TESTE RANDOM FOREST REGRESSOR (DIABETES Dataset) " + "="*10)
    
    # 1. Carregar e Preparar Dados
    data = load_diabetes()
    X, y = data.data, data.target
    
    # Dividir dados (Treino/Teste)
    # Não precisamos de StandardScaler aqui, pois árvores de decisão são robustas a escala
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # ===============================================
    # 2. REFERÊNCIA 1: Sua Árvore Simples (Base)
    # ===============================================
    print("\n--- 🌲 Referência: Sua DecisionTreeRegressor (1 Árvore) ---")
    
    # Usamos max_depth=4 (como no seu teste anterior)
    dtr_base = ManualDTR(max_depth=4, criterion='mse')
    
    # A Árvore de Regressão original não usa max_features, então passamos None
    # NOTA: Se você ainda não adaptou o DTR para max_features, ele vai ignorar o parâmetro
    # mas rodará se você passar None ou nada.
    dtr_base.fit(X_train, y_train)
    
    # Predições e Métricas
    y_pred_base = dtr_base.predict(X_test)
    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)
    
    print(f"MSE (Árvore Simples Manual): {mse_base:.2f}")
    print(f"R² (Árvore Simples Manual): {r2_base:.4f}")

    # ===============================================
    # 3. TESTE: Seu RandomForestRegressor
    # ===============================================
    print("\n--- 🌳 Seu RandomForestRegressor (20 Árvores) ---")
    
    # Parâmetros: n_trees=20 (Bom equilíbrio) e max_depth=7 (Permitimos mais crescimento)
    # max_features='sqrt' (Padrão para RF)
    rfc_manual = ManualRFR(n_trees=20, max_depth=7, max_features='sqrt')
    
    # O treinamento de 20 árvores será visivelmente mais lento
    rfc_manual.fit(X_train, y_train) 
    
    # Predições e Métricas
    y_pred_manual = rfc_manual.predict(X_test)
    mse_manual = mean_squared_error(y_test, y_pred_manual)
    r2_manual = r2_score(y_test, y_pred_manual)
    
    print(f"MSE (Random Forest Manual): {mse_manual:.2f}")
    print(f"R² (Random Forest Manual): {r2_manual:.4f}")

    # ===============================================
    # 4. REFERÊNCIA 2: Sklearn RandomForestRegressor
    # ===============================================
    print("\n--- 🚀 Sklearn RandomForestRegressor (para validação) ---")
    
    # Usamos os mesmos parâmetros para comparação justa:
    rfr_sklearn = SklearnRFR(n_estimators=20, max_depth=7, max_features='sqrt', random_state=42)
    rfr_sklearn.fit(X_train, y_train)
    
    # Predições e Métricas
    y_pred_sklearn = rfr_sklearn.predict(X_test)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    
    print(f"MSE (Random Forest Sklearn): {mse_sklearn:.2f}")
    print(f"R² (Random Forest Sklearn): {r2_sklearn:.4f}")

    # ===============================================
    # 5. RESUMO FINAL
    # ===============================================
    print("\n" + "="*80)
    print(f"| Resumo de Regressão (R²) | Base DTR: {r2_base:.4f} | RF Manual: {r2_manual:.4f} | RF Sklearn: {r2_sklearn:.4f} |")
    print("="*80)

# test_regressor.py (Parte final)

if __name__ == "__main__":
    test_random_forest_regressor(DecisionTreeRegressor, RandomForestRegressor) 
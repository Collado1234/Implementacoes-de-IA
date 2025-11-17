# test_random_forest.py

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Importa o Random Forest do Sklearn para comparação
from sklearn.ensemble import RandomForestClassifier as SklearnRFC
from random_forest_classifier import RandomForestClassifier
from decision_tree_classifier import DecisionTreeClassifier
# Assumindo que você tem:
# from decision_tree_classifier import DecisionTreeClassifier
# from random_forest import RandomForestClassifier 
# (Se as classes estiverem no mesmo arquivo, você não precisa destes imports)

# =============================================================
# MÓDULO DE TESTE: RANDOM FOREST (IRIS Dataset)
# =============================================================

def test_random_forest_classification():
    print("="*15 + " TESTE RANDOM FOREST (IRIS Dataset) " + "="*15)
    
    # 1. Carregar Dados e Dividir
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    # Usamos um random_state para reprodutibilidade
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ===============================================
    # 2. REFERÊNCIA 1: Sua Árvore Simples (Base)
    # ===============================================
    print("\n--- 🌲 Referência: Sua DecisionTreeClassifier (1 Árvore) ---")
    
    # Profundidade 3 (onde você tinha 95.56% antes)
    dtc_base = DecisionTreeClassifier(max_depth=3, criterion='gini')
    dtc_base.fit(X_train, y_train)
    y_pred_base = dtc_base.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    
    print(f"Acurácia da Árvore Simples (Manual): {acc_base:.4f}")

    # ===============================================
    # 3. TESTE: Seu RandomForestClassifier
    # ===============================================
    print("\n--- 🌳 Seu RandomForestClassifier (10 Árvores) ---")
    
    # Parâmetros: 
    # n_trees=10 (Um número pequeno para teste rápido)
    # max_features='sqrt' (Padrão para RF)
    # max_depth=5 (Permite que as árvores cresçam um pouco mais, pois o bagging 
    #              irá compensar o overfitting de cada árvore individual)
    
    rfc_manual = RandomForestClassifier(n_trees=10, max_depth=5, max_features='sqrt')
    rfc_manual.fit(X_train, y_train)
    
    y_pred_manual = rfc_manual.predict(X_test)
    acc_manual = accuracy_score(y_test, y_pred_manual)
    
    print(f"Acurácia do Random Forest (Manual): {acc_manual:.4f}")
    print("\nRelatório de Classificação (Seu Código):\n")
    print(classification_report(y_test, y_pred_manual, target_names=target_names, zero_division=0))

    # ===============================================
    # 4. REFERÊNCIA 2: Sklearn RandomForestClassifier
    # ===============================================
    print("\n--- 🚀 Sklearn RandomForestClassifier (para validação) ---")
    
    # Sklearn usa max_features='sqrt' por padrão e max_depth=None (full growth)
    # Vamos usar os mesmos parâmetros para comparação justa:
    rfc_sklearn = SklearnRFC(n_estimators=10, max_depth=5, max_features='sqrt', random_state=42)
    rfc_sklearn.fit(X_train, y_train)
    
    y_pred_sklearn = rfc_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"Acurácia do Random Forest (Sklearn): {acc_sklearn:.4f}")

    # ===============================================
    # 5. RESUMO
    # ===============================================
    print("\n" + "="*70)
    print("| RESUMO DE ACURÁCIA |")
    print(f"| Árvore Simples (Manual): {acc_base:.4f} |")
    print(f"| Random Forest (Manual): {acc_manual:.4f} |")
    print(f"| Random Forest (Sklearn): {acc_sklearn:.4f} |")
    print("="*70)

def test_random_forest_breast_cancer():
    print("="*10 + " TESTE RANDOM FOREST (BREAST CANCER) " + "="*10)
    
    # 1. Carregar Dados e Dividir
    data = load_breast_cancer()
    X = data.data
    y = data.target
    target_names = data.target_names # 'malignant' e 'benign'
    
    # Dividir dados (Treino/Teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ===============================================
    # 2. REFERÊNCIA 1: Sua Árvore Simples (Base)
    # ===============================================
    print("\n--- 🌲 Referência: Sua DecisionTreeClassifier (1 Árvore) ---")
    
    # Permitimos que a árvore cresça mais para ver sua acurácia máxima isolada
    dtc_base = DecisionTreeClassifier(max_depth=None, criterion='gini') 
    dtc_base.fit(X_train, y_train)
    y_pred_base = dtc_base.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    
    print(f"Acurácia da Árvore Simples (Manual): {acc_base:.4f}")

    # ===============================================
    # 3. TESTE: Seu RandomForestClassifier
    # ===============================================
    print("\n--- 🌳 Seu RandomForestClassifier (50 Árvores) ---")
    
    # Aumentamos para 50 árvores para um teste mais robusto e próximo do real
    # Usamos max_depth=10 e max_features='sqrt'
    rfc_manual = RandomForestClassifier(n_trees=50, max_depth=10, max_features='sqrt')
    
    # O treinamento de 50 árvores será visivelmente mais lento, o que é esperado
    rfc_manual.fit(X_train, y_train) 
    
    y_pred_manual = rfc_manual.predict(X_test)
    acc_manual = accuracy_score(y_test, y_pred_manual)
    
    print(f"Acurácia do Random Forest (Manual): {acc_manual:.4f}")
    print("\nRelatório de Classificação (Seu Código):\n")
    print(classification_report(y_test, y_pred_manual, target_names=target_names, zero_division=0))

    # ===============================================
    # 4. REFERÊNCIA 2: Sklearn RandomForestClassifier
    # ===============================================
    print("\n--- 🚀 Sklearn RandomForestClassifier (para validação) ---")
    
    rfc_sklearn = SklearnRFC(n_estimators=50, max_depth=10, max_features='sqrt', random_state=42)
    rfc_sklearn.fit(X_train, y_train)
    
    y_pred_sklearn = rfc_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"Acurácia do Random Forest (Sklearn): {acc_sklearn:.4f}")

    # ===============================================
    # 5. RESUMO
    # ===============================================
    print("\n" + "="*70)
    print("| RESUMO DE ACURÁCIA (BREAST CANCER) |")
    print(f"| Árvore Simples (Manual): {acc_base:.4f} |")
    print(f"| Random Forest (Manual): {acc_manual:.4f} |")
    print(f"| Random Forest (Sklearn): {acc_sklearn:.4f} |")
    print("="*70)

if __name__ == "__main__":
    test_random_forest_classification()
    test_random_forest_breast_cancer()
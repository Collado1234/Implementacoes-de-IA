# main_logistic.py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression as SklearnLogisticReg
from logist_regression import LogisticReg

# Importa sua classe (supondo que esteja no mesmo arquivo ou em 'logistic_reg.py')
# Se a classe LogisticReg estiver neste mesmo arquivo, remova a linha abaixo.
# from logistic_reg import LogisticReg 
# ... (Se a classe LogisticReg estiver no início deste arquivo, mantenha-a)

# ... (Se a classe LogisticReg estiver em outro arquivo, mantenha este import)
# from LogisticReg import LogisticReg


# =============================================================
# FUNÇÃO PRINCIPAL DE TESTE
# =============================================================

def test_logistic_regression(ManualLogisticReg):
    print("="*10 + " TESTE REGRESSÃO LOGÍSTICA (BREAST CANCER) " + "="*10)
    
    # 1. Carregar e Preparar Dados
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # É uma boa prática escalar dados para Regressão Logística com GD
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir dados (Treino/Teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # --- 2. Seu Modelo (Implementação Manual) ---
    print("\n--- 🌳 Sua Implementação Manual de Regressão Logística ---")
    
    # Usamos uma taxa de aprendizado e iterações que funcionam bem para este dataset escalado
    logreg_manual = ManualLogisticReg(learning_rate=0.01, iterations=5000)
    
    # Nota: Seu 'fit' espera que 'y' seja um vetor numpy
    logreg_manual.fit(X_train, y_train)
    
    # Predições
    y_pred_manual = logreg_manual.predict(X_test)
    proba_manual = logreg_manual.predict_proba(X_test)
    
    # Métricas
    accuracy_manual = accuracy_score(y_test, y_pred_manual)
    
    print(f"Acurácia no Teste: {accuracy_manual:.4f}")
    print("\nRelatório de Classificação (Seu Código):\n")
    # O classification_report é ótimo para ver Precision, Recall e F1
    print(classification_report(y_test, y_pred_manual, target_names=['Maligno (0)', 'Benigno (1)']))
    
    # --- 3. Modelo Sklearn para Comparação ---
    print("\n--- 🚀 Sklearn LogisticRegression (para validação) ---")
    
    # Sklearn usa otimizadores mais avançados (ex: 'lbfgs') e regularização (C=1.0)
    logreg_sklearn = SklearnLogisticReg(solver='liblinear', C=1.0, random_state=42, max_iter=5000)
    logreg_sklearn.fit(X_train, y_train)
    
    # Predições
    y_pred_sklearn = logreg_sklearn.predict(X_test)
    
    # Métricas
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"Sklearn Acurácia no Teste: {accuracy_sklearn:.4f}")
    print("\nRelatório de Classificação (Sklearn):\n")
    print(classification_report(y_test, y_pred_sklearn, target_names=['Maligno (0)', 'Benigno (1)']))
    
    # --- 4. Comparação Final ---
    print("\n" + "="*55)
    print(f"| Comparação de Acurácia | Manual: {accuracy_manual:.4f} | Sklearn: {accuracy_sklearn:.4f} |")
    print("="*55)


if __name__ == "__main__":
    # Supondo que a classe LogisticReg está definida no topo deste arquivo ou importada.
    test_logistic_regression(LogisticReg)
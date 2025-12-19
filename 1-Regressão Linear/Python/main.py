import numpy as np
from sklearn.datasets import load_diabetes
from regressao_linear import MyLinearRegresssion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

def main():
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    Y = diabetes.target

    # Escolher uma feature para visualização
    feature = 'bmi'
    X_feat = X[[feature]]

    X_train, X_test, Y_train, Y_test = train_test_split(X_feat, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    methods = [
        ('exact', MyLinearRegresssion(method='exact')),
        ('gd', MyLinearRegresssion(method='gd', learning_rate=0.01, n_iterations=10000)),
        ('ridge', MyLinearRegresssion(method='ridge', lambda_rate=1.0, learning_rate=0.01, n_iterations=10000)),
        ('lasso', MyLinearRegresssion(method='lasso', lambda_rate=1.0, learning_rate=0.01, n_iterations=10000)),
        ('elasticnet', MyLinearRegresssion(method='elasticnet', lambda_rate=1.0, alpha_rate=0.5, learning_rate=0.01, n_iterations=10000))
    ]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, Y_test, color='gray', label='Dados reais', alpha=0.6)

    # Gera pontos ordenados para desenhar as retas
    X_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)

    for name, model in methods:
        model.model.fit(X_train, Y_train)
        Y_pred_line = model.model.predict(X_line)

        plt.plot(X_line, Y_pred_line, label=f'{name}', linewidth=2)

        # Avaliação numérica
        Y_pred = model.model.predict(X_test)
        mae = mean_absolute_error(Y_test, Y_pred)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    plt.title(f"Comparação das Retas de Regressão ({feature})")
    plt.xlabel(feature)
    plt.ylabel("Progresso da doença (target)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

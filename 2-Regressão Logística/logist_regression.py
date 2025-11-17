import numpy as np

class LogisticReg:
    """
    Uma implementação simples de Regressão Logística para classificação binária.
    
    A regressão logística é um modelo estatístico usado para prever a probabilidade de um 
    evento de dois resultados (classe 0 ou classe 1) com base em variáveis independentes.

    A classe LogisticReg implementa o algoritmo de otimização de gradiente descendente para
    aprender os parâmetros do modelo (theta) a partir dos dados de treinamento.
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Inicializa os parâmetros do modelo de Regressão Logística.
        
        Parâmetros:
        learning_rate (float): Taxa de aprendizado do gradiente descendente. Controla o tamanho 
                               do passo durante a atualização dos parâmetros.
        iterations (int): Número de iterações (ou épocas) do processo de otimização.
        """
        self.intercept = None  # Interceptação (termo bias) do modelo
        self.theta = None       # Coeficientes (pesos) do modelo
        self.learning_rate = learning_rate  # Taxa de aprendizado
        self.iterations = iterations  # Número de iterações

    def sigmoid(self, z):
        """
        Função sigmoide que transforma um valor real em um valor entre 0 e 1.
        
        A função sigmoide é usada para modelar a probabilidade de um evento ocorrer.
        
        Parâmetros:
        z (ndarray): Vetor ou valor escalar para o qual a função sigmoide será aplicada.

        Retorna:
        float ou ndarray: Valor(s) transformado(s) pela função sigmoide.
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Treina o modelo de Regressão Logística usando o método de Gradiente Descendente.
        
        Parâmetros:
        X (ndarray): Matriz de características (tamanho: n amostras x m características).
        y (ndarray): Vetor de rótulos (tamanho: n amostras) contendo valores binários (0 ou 1).
        
        O método ajusta os parâmetros (theta) do modelo de modo a minimizar o erro de previsão.
        """
        # Adiciona uma coluna de 1s para o termo de interceptação (bias)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Inicializa os coeficientes theta com zeros
        m, n = Xb.shape
        self.theta = np.zeros(n)

        # Gradiente descendente para otimização dos parâmetros
        for _ in range(self.iterations):
            # Previsões usando a função sigmoide
            predictions = self.sigmoid(Xb.dot(self.theta)) # Fórmula Predictions: sigmoid(X * theta)
            
            # Erros (diferença entre previsões e rótulos reais)
            errors = predictions - y
            
            # Gradiente (derivada do erro com relação aos parâmetros)
            gradient = (1 / m) * Xb.T.dot(errors)  # Fórmula Gradiente: (1/m) * X^T * (erro)
            
            # Atualiza os parâmetros (theta)
            self.theta -= self.learning_rate * gradient # Fórmula Atualização: theta = theta - learning_rate * gradiente

        # O intercepto é o primeiro valor de theta, os demais são os coeficientes
        self.intercept = self.theta[0]
        self.theta = self.theta[1:]

    def predict_proba(self, X):
        """
        Preve a probabilidade de cada amostra pertencer à classe 1 (probabilidade logística).
        
        Parâmetros:
        X (ndarray): Matriz de características (tamanho: n amostras x m características).
        
        Retorna:
        ndarray: Vetor de probabilidades de cada amostra pertencer à classe 1.
        """
        Xb = np.c_[np.ones((X.shape[0], 1)), X]  # Adiciona coluna de 1s para o bias
        return self.sigmoid(Xb.dot(np.r_[self.intercept, self.theta]))  # Calcula a probabilidade

    def predict(self, X, threshold=0.5):
        """
        Preve a classe (0 ou 1) para cada amostra com base em um limiar de probabilidade.
        
        Parâmetros:
        X (ndarray): Matriz de características (tamanho: n amostras x m características).
        threshold (float): Limiar de decisão para determinar a classe. 
                           Se a probabilidade for maior ou igual a este valor, a classe será 1, caso contrário será 0.
        
        Retorna:
        ndarray: Vetor com as classes preditas (0 ou 1) para cada amostra.
        """
        probabilities = self.predict_proba(X)  # Obtém as probabilidades
        return (probabilities >= threshold).astype(int)  # Aplica o limiar e retorna 0 ou 1

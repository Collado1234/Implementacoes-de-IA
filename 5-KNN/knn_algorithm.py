# knn_algorithm_vectorized.py
"""
Implementação do KNN (Classificação e Regressão) totalmente vetorizada em NumPy.
Muito mais rápida para datasets grandes.
"""

import numpy as np
from collections import Counter

def get_neighbors_vectorized(X_train, y_train, X_test_sample, k):
    """
    Calcula as distâncias entre uma amostra de teste e todos os pontos de treino
    de forma vetorizada e retorna os rótulos ou valores dos K vizinhos mais próximos.
    """
    # Subtração vetorizada e soma ao quadrado por linha
    distances = np.sqrt(np.sum((X_train - X_test_sample) ** 2, axis=1))
    
    # Índices dos K menores valores (mais próximos)
    neighbors_idx = np.argpartition(distances, k)[:k]  # mais rápido que argsort completo
    
    # Retorna os rótulos/valores dos vizinhos
    return y_train[neighbors_idx]

def knn_classifier_vectorized(X_train, y_train, X_test, k):
    """
    Classifica amostras de teste usando votação majoritária entre K vizinhos mais próximos.
    """
    predictions = []

    for x_test_sample in X_test:
        neighbors_labels = get_neighbors_vectorized(X_train, y_train, x_test_sample, k)
        most_common = Counter(neighbors_labels).most_common(1)
        predictions.append(most_common[0][0])

    return np.array(predictions)


def knn_regressor_vectorized(X_train, y_train, X_test, k):
    """
    Prediz valores para amostras de teste usando a média dos K vizinhos mais próximos.
    """
    predictions = []

    for x_test_sample in X_test:
        neighbors_values = get_neighbors_vectorized(X_train, y_train, x_test_sample, k)
        prediction = np.mean(neighbors_values)
        predictions.append(prediction)

    return np.array(predictions) 

import numpy as np
from typing import Optional

class KMeans:
    """
    Implementação aprimorada do algoritmo K-Means com inicialização K-Means++ e vetorização.
    """

    def __init__(self, 
                 K: int, 
                 max_iter: int = 100, 
                 tolerance: float = 1e-4, 
                 distance_metric: str = 'euclidean',
                 random_state: Optional[int] = None):
        """
        :param K: Número de clusters (grupos) a serem formados.
        :param max_iter: Número máximo de iterações.
        :param tolerance: Tolerância para a convergência.
        :param distance_metric: Métrica de distância ('euclidean' atualmente).
        :param random_state: Semente para reprodutibilidade.
        """
        self.K = K
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.random_state = random_state

        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

        if random_state is not None:
            np.random.seed(random_state)

    # ----------------------------------------------------
    # INICIALIZAÇÃO K-MEANS++
    # ----------------------------------------------------
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Inicializa os centroides usando o método K-Means++.
        Esse método tende a gerar convergência mais rápida e melhor distribuição inicial.
        """
        n_samples = X.shape[0]
        centroids = []

        # 1. Escolhe o primeiro centroide aleatoriamente
        first_idx = np.random.choice(n_samples)
        centroids.append(X[first_idx])

        # 2. Escolhe os demais centroides com probabilidade proporcional à distância ao quadrado
        for _ in range(1, self.K):
            # Calcula distâncias ao centroide mais próximo
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2) ** 2, axis=1)
            # Probabilidade proporcional à distância²
            probabilities = distances / np.sum(distances)
            # Escolhe o próximo centroide
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx])

        return np.array(centroids)

    # ----------------------------------------------------
    # ATRIBUIÇÃO DE CLUSTERS (E-STEP)
    # ----------------------------------------------------
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Atribui cada ponto ao centroide mais próximo (versão vetorizada)."""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    # ----------------------------------------------------
    # ATUALIZAÇÃO DOS CENTROIDES (M-STEP)
    # ----------------------------------------------------
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recalcula os centroides como a média dos pontos em cada cluster."""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Se o cluster ficou vazio, mantemos o centroide antigo
                new_centroids[k] = self.centroids[k]
        return new_centroids

    # ----------------------------------------------------
    # TREINAMENTO DO MODELO (FIT)
    # ----------------------------------------------------
    def fit(self, X: np.ndarray):
        """
        Executa o algoritmo K-Means até convergir ou atingir o número máximo de iterações.
        """
        X = np.asarray(X)
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()

            # E-Step
            labels = self._assign_clusters(X)

            # M-Step
            self.centroids = self._update_centroids(X, labels)

            # Critério de convergência
            centroid_movement = np.mean(np.linalg.norm(self.centroids - old_centroids, axis=1))
            print(f"Iteração {i+1}: movimento médio dos centroides = {centroid_movement:.6f}")

            if centroid_movement < self.tolerance:
                print(f"K-Means convergiu após {i+1} iterações.")
                break

        self.labels = self._assign_clusters(X)
        print("Treinamento K-Means concluído.")

    # ----------------------------------------------------
    # PREDIÇÃO DE NOVOS PONTOS
    # ----------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Atribui novos pontos aos clusters existentes."""
        if self.centroids is None:
            raise ValueError("O modelo precisa ser treinado antes de prever.")
        return self._assign_clusters(np.asarray(X))

    # ----------------------------------------------------
    # OBTÉM RÓTULOS FINAIS
    # ----------------------------------------------------
    def get_labels(self) -> np.ndarray:
        """Retorna os rótulos de cluster do conjunto de treino."""
        if self.labels is None:
            raise ValueError("O modelo K-Means não foi treinado.")
        return self.labels

    # ----------------------------------------------------
    # OBTÉM CENTROIDES
    # ----------------------------------------------------
    def get_centroids(self) -> np.ndarray:
        """Retorna os centroides atuais."""
        if self.centroids is None:
            raise ValueError("O modelo K-Means não foi treinado.")
        return self.centroids


# ----------------------------------------------------
# TESTE DE FUNCIONAMENTO
# ----------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Gera dados sintéticos
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

    model = KMeans(K=3, random_state=42)
    model.fit(X)

    centroids = model.get_centroids()
    labels = model.get_labels()

    # Visualização
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
    plt.title("Clusters encontrados pelo K-Means++")
    plt.show()

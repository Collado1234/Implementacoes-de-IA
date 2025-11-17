import numpy as np
from typing import Optional

class KMeans:
    """
    Implementação do algoritmo K-Means para agrupamento (Clustering).
    """
    def __init__(self, K: int, max_iter: int = 100, tolerance: float = 1e-4, distance_metric: str = 'euclidean'):
        """
        :param K: O número de clusters (grupos) a serem formados.
        :param max_iter: Número máximo de iterações.
        :param tolerance: Tolerância para a convergência (mudança mínima no centroide).
        :param distance_metric: Métrica de distância a ser usada (por enquanto, apenas 'euclidean').
        """
        self.K = K
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None # Rótulos de cluster para cada ponto

    # ----------------------------------------------------
    # MÉTODOS AUXILIARES
    # ----------------------------------------------------

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Inicializa K centroides selecionando K pontos aleatórios do dataset."""
        n_samples = X.shape[0]
        # Seleciona K índices únicos aleatoriamente
        random_indices = np.random.choice(n_samples, size=self.K, replace=False)
        # Os centroides iniciais são os pontos de dados nesses índices
        initial_centroids = X[random_indices]
        return initial_centroids

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calcula a distância entre dois pontos (por enquanto, apenas Euclidiana)."""
        # Distância Euclidiana ao quadrado (para otimização, evitamos a raiz quadrada,
        # pois a atribuição de cluster depende apenas da ordem das distâncias).
        if self.distance_metric == 'euclidean':
            return np.sum((p1 - p2) ** 2)
        # Futuramente, você pode adicionar outras métricas aqui.
        else:
             raise ValueError("Métrica de distância não suportada.")


    # ----------------------------------------------------
    # PASSO 1 DO LOOP: ATRIBUIÇÃO (Expectation)
    # ----------------------------------------------------
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Atribui cada ponto de dado ao centroide mais próximo."""
        n_samples = X.shape[0]
        # Inicializa um array para armazenar o rótulo (índice do centroide) de cada ponto
        labels = np.zeros(n_samples, dtype=int) 
        
        for i, point in enumerate(X):
            min_dist = float('inf')
            closest_centroid_index = -1
            
            # Compara o ponto i com todos os K centroides
            for k in range(self.K):
                distance = self._distance(point, self.centroids[k])
                
                if distance < min_dist:
                    min_dist = distance
                    closest_centroid_index = k
            
            # Atribui o ponto ao centroide mais próximo
            labels[i] = closest_centroid_index
            
        return labels

    # ----------------------------------------------------
    # PASSO 2 DO LOOP: ATUALIZAÇÃO (Maximization)
    # ----------------------------------------------------
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recalcula os centroides como a média dos pontos em cada cluster."""
        new_centroids = np.zeros_like(self.centroids) # Cria uma matriz do mesmo formato
        
        for k in range(self.K):
            # Encontra todos os pontos atribuídos ao cluster k
            points_in_cluster = X[labels == k]
            
            if len(points_in_cluster) > 0:
                # O novo centroide é a média dos pontos no cluster
                new_centroids[k] = np.mean(points_in_cluster, axis=0)
            else:
                # Se um cluster estiver vazio (raro, mas possível), 
                # mantemos o centroide antigo ou o reinicializamos. 
                # Vamos manter o antigo por simplicidade.
                new_centroids[k] = self.centroids[k]
                
        return new_centroids
    
    # k_means.py (continuação)

    def fit(self, X: np.ndarray):
        """
        Executa o algoritmo K-Means.

        O algoritmo itera entre atribuição e atualização até a convergência
        ou o número máximo de iterações ser atingido.
        """
        X = np.asarray(X)
        
        # 1. Inicialização
        self.centroids = self._initialize_centroids(X)
        
        for i in range(self.max_iter):
            # Salva os centroides antigos para verificar a convergência
            old_centroids = self.centroids.copy()
            
            # E-Step (Expectation/Atribuição)
            labels = self._assign_clusters(X)
            
            # M-Step (Maximization/Atualização)
            self.centroids = self._update_centroids(X, labels)
            
            # Verifica Convergência: A convergência ocorre quando a mudança
            # na posição dos centroides está abaixo da tolerância.
            centroid_movement = np.sum(np.linalg.norm(self.centroids - old_centroids, axis=1))
            
            print(f"Iteração {i+1}: Movimento total do centroide: {centroid_movement:.4f}")
            
            if centroid_movement < self.tolerance:
                print(f"K-Means convergiu após {i+1} iterações.")
                break
        
        self.labels = self._assign_clusters(X) # Atribuição final dos rótulos
        print("Treinamento K-Means concluído.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Atribui novos pontos de dado aos clusters existentes (centroides).
        Isto é, essencialmente, o passo de atribuição.
        """
        X = np.asarray(X)
        if self.centroids is None:
            raise ValueError("O modelo K-Means deve ser treinado antes de prever.")
            
        return self._assign_clusters(X)

    def get_labels(self) -> np.ndarray:
        """Retorna os rótulos de cluster do conjunto de dados de treino."""
        if self.labels is None:
             raise ValueError("O modelo K-Means não foi treinado.")
        return self.labels
    

# k_means.py (continuação)


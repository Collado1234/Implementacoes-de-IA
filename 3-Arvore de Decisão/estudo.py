import numpy as np
from collections import Counter

# --- Funções Auxiliares de Impureza ---

def entropy(y):
    """Calcula a Entropia de um conjunto de rótulos (y)."""
    # Conta a frequência de cada classe
    hist = np.bincount(y) 
    # Calcula a probabilidade P_i de cada classe
    ps = hist / len(y)
    
    # Entropia = - Sum(P_i * log2(P_i))
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def information_gain(y_pai, y_esq, y_dir):
    """Calcula o Ganho de Informação da divisão."""
    
    # Entropia Ponderada (Após)
    p_esq = len(y_esq) / len(y_pai)
    entropia_ponderada = p_esq * entropy(y_esq) + (1 - p_esq) * entropy(y_dir)
    
    # Ganho = Entropia Antes - Entropia Ponderada Depois
    return entropy(y_pai) - entropia_ponderada

# --- Estrutura do Nó ---

class Node:
    """Representa um nó na árvore (interno ou folha)."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        # Para nós de decisão
        self.feature_idx = feature_idx  # Índice da feature para split
        self.threshold = threshold      # Ponto de corte
        self.left = left                # Subárvore esquerda
        self.right = right              # Subárvore direita
        # Para nós folha
        self.value = value              # Classe predita (Moda)

    def is_leaf(self):
        return self.value is not None

# --- Classificador ---

class SimpleDecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def _best_split(self, X, y):
        """Encontra o split que maximiza o Ganho de Informação."""
        m, n = X.shape
        best_gain = -1.0
        best_idx, best_thr = None, None

        # Itera sobre todas as features (colunas)
        for idx in range(n):
            X_column = X[:, idx]
            thresholds = np.unique(X_column) # Valores únicos como potenciais thresholds

            # Itera sobre todos os thresholds possíveis
            for thr in thresholds:
                # Divide o conjunto de dados (simplesmente comparando)
                indices_esq = X_column < thr
                y_esq = y[indices_esq]
                y_dir = y[~indices_esq]

                # Ignora splits que criam nós vazios
                if len(y_esq) == 0 or len(y_dir) == 0:
                    continue

                # Calcula o Ganho de Informação para este split
                gain = information_gain(y, y_esq, y_dir)

                # Atualiza o melhor split
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr, best_gain

    def _build_tree(self, X, y, depth):
        """Função recursiva para construir a árvore."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 1. Determina a classe majoritária (Moda) para o nó atual
        most_common = Counter(y).most_common(1)[0][0]

        # 2. Condições de Parada (Nó Folha)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < 2):
            return Node(value=most_common)

        # 3. Encontra o Melhor Split
        feature_idx, threshold, gain = self._best_split(X, y)

        # 4. Condição de Parada (Sem Ganho)
        # Se o melhor ganho for muito baixo, paramos e criamos uma folha.
        if gain < 1e-6: 
             return Node(value=most_common)
        
        # 5. Split (Criação do Nó Interno)
        
        # Divide os dados de forma definitiva usando o melhor split
        X_column = X[:, feature_idx]
        indices_esq = X_column < threshold
        
        X_esq, y_esq = X[indices_esq], y[indices_esq]
        X_dir, y_dir = X[~indices_esq], y[~indices_esq]
        
        # 6. Chamadas Recursivas (Constrói Subárvores)
        left_child = self._build_tree(X_esq, y_esq, depth + 1)
        right_child = self._build_tree(X_dir, y_dir, depth + 1)
        
        # Retorna o Nó de Decisão
        return Node(feature_idx, threshold, left_child, right_child)

    def fit(self, X, y):
        """Inicia o treinamento da árvore."""
        self.root = self._build_tree(X, y, depth=0)

    def _traverse_tree(self, x, node):
        """Atravessa a árvore para fazer a previsão de uma única amostra."""
        if node.is_leaf():
            return node.value
        
        # Regra de Decisão
        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """Faz previsões para um conjunto de amostras."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

# Exemplo de Uso (Teste)
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Carregando dados de exemplo (Iris)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinando o modelo
    model = SimpleDecisionTree(max_depth=5)
    model.fit(X_train, y_train)
    
    # Fazendo previsões
    y_pred = model.predict(X_test)
    
    # Calculando Acurácia (Accuracy)
    accuracy = np.mean(y_pred == y_test)
    print(f"Acurácia: {accuracy:.4f}")
    
    # Para visualizar as regras de decisão, use a função print_tree
    # que você já viu em exemplos mais completos.
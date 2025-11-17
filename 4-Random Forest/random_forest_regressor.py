
import numpy as np
from decision_tree_regressor import DecisionTreeRegressor 

class RandomForestRegressor:
    """
    An implementation of the Random Forest algorithm for regression tasks.
    This class creates an ensemble of DecisionTreeRegressor trees, each trained
    """
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, 
                 max_features='sqrt', criterion='mse'): # <-- 'mse' is the common pattern for regression
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        
        # O critério de regressão é fixo como 'mse' (Mean Squared Error)
        assert criterion == 'mse', "Criterion for Regressor should be 'mse' (Mean Squared Error)"
        self.criterion = criterion 
        
        self.trees = [] 
        self.n_features = None 
        self.n_features_per_split = None 
    
    def _get_bootstrap_sample(self, X, y):
        """Cria um subconjunto de dados de treino usando amostragem com reposição (Bootstrap)."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _get_feature_subset_size(self, n_total_features):
        """Calcula o número de features a serem consideradas em cada split."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_total_features

    # random_forest_regressor.py (continuação)

    def fit(self, X, y):
        """
        Treina o Random Forest Regressor. Cria e treina n_trees árvores independentemente.
        """
        self.n_features = X.shape[1]
        self.n_features_per_split = self._get_feature_subset_size(self.n_features)
        self.trees = [] 
        
        for _ in range(self.n_trees):
            
            X_sample, y_sample = self._get_bootstrap_sample(X, y)
            
            # Diferença Chave: Usamos DecisionTreeRegressor
            tree = DecisionTreeRegressor( 
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion, # Será 'mse'
                max_features=self.n_features_per_split 
            )
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        print(f"Treinamento concluído. {self.n_trees} árvores de regressão treinadas.")

    def predict(self, X):
        """
        Faz a predição agregando as previsões de todas as árvores por média.

        Retorna:
        ndarray: O vetor de valores preditos (média das predições).
        """
        # Coleta as predições de todas as árvores
        # predictions será uma matriz onde cada coluna é a predição de uma árvore.
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # 2. Agregação por Média (Média Simples)
        # Diferença Chave: Em vez de voto, tiramos a média ao longo do eixo das árvores (axis=0)
        y_pred = np.mean(predictions, axis=0)
        
        return y_pred
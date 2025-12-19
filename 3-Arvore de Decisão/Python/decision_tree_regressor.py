import numpy as np
from collections import Counter
from typing import Optional, Tuple, Dict

def mean_squared_error_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Erro Quadrático Médio (MSE) manualmente."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays y_true e y_pred devem ter o mesmo tamanho.")
        
    m = y_true.size
    # (y_true - y_pred) ** 2 calcula o erro ao quadrado para cada ponto
    squared_errors = (y_true - y_pred) ** 2
    
    # np.sum soma todos os erros, e dividimos por m (o número de amostras)
    mse = np.sum(squared_errors) / m
    return mse

# NOTA: Esta classe DecisionTreeNode foi mantida, mas a classe DecisionTreeRegressor
# NÃO a usa completamente, pois a regressão não precisa de num_samples_per_class ou predicted_class (que armazena um float).
# Para fins de organização, o nome 'gini' poderia ser mudado para 'impurity', mas vamos manter
# como está para evitar mudanças grandes, já que 'predicted_class' armazena o valor de predição (float) na regressão.
class DecisionTreeNode:
    def __init__(self,
                 gini: float = None,
                 num_samples: int = None,
                 num_samples_per_class: Optional[np.ndarray] = None,
                 predicted_class: Optional[float] = None, # Alterei para Optional[float] para clareza em regressor
                 feature_index: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional['DecisionTreeNode'] = None,
                 right: Optional['DecisionTreeNode'] = None):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right 

    def is_leaf(self):
        """Retorna True se este nó for uma folha (sem filhos)."""
        return self.left is None and self.right is None 

class DecisionTreeRegressor():
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 criterion='mse',
                 min_impurity_decrease=0.0):
        assert criterion in ['mse','mae'], "Criterion must be 'mse' or 'mae'"
        self.max_depth = max_depth
        # CORREÇÃO 1: 'self,min_samples_split' -> 'self.min_samples_split'
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.n_features_ = None
        self.feature_importances_ = None
        self.root: Optional[DecisionTreeNode] = None

    def _mse(self, y:np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def _mae(self, y:np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        median = np.median(y)
        return np.mean(np.abs(y - median))
    
    def _impurity(self, y:np.ndarray) -> float:
        if self.criterion == 'mse':
            return self._mse(y)
        else:
            return self._mae(y)
    
    def _accumulate_feature_importance(self, idx: int, gain: float):
        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        self.feature_importances_[idx] += gain

    def _best_split(self, X:np.ndarray, y:np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        m, n = X.shape
        if m < self.min_samples_split:
            return None , None, 0.0
        
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_idx, best_thr = None, None

        for feature_index in range(n):
            vals = X[:, feature_index]
            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            sorted_y = y[sorted_idx]

            for i in range (1,m):
                #avoid thresholds with the same value
                if sorted_vals[i] == sorted_vals[i-1]:
                    continue
                thr = (sorted_vals[i] + sorted_vals[i-1]) / 2.0

                y_left = sorted_y[:i]
                y_right = sorted_y[i:]

                impurity_left = self._impurity(y_left)
                impurity_right = self._impurity(y_right)

                n_left = y_left.size
                n_right = y_right.size

                weighted_impurity = (n_left * impurity_left + n_right * impurity_right) / m

                info_gain = parent_impurity - weighted_impurity

                if info_gain > best_gain + 1e-12:
                    best_gain = info_gain
                    best_idx = feature_index
                    best_thr = thr
                    
        return best_idx, best_thr, best_gain

    def _build_tree(self, X:np.ndarray, y:np.ndarray, depth:int=0) -> DecisionTreeNode:
            predicted_value = np.mean(y) 

            node = DecisionTreeNode(
                gini = self._impurity(y),
                num_samples=y.size,
                num_samples_per_class=None,
                predicted_class=predicted_value
            )

            if (self.max_depth is not None and depth >= self.max_depth) or y.size < self.min_samples_split or np.var(y) < 1e-8:
                return node
            
            idx, thr, gain = self._best_split(X, y)

            if idx is None or gain <= self.min_impurity_decrease:
                return node
            
            indices_left = X[:,idx] < thr
            left = self._build_tree(X[indices_left], y[indices_left], depth + 1)
            right = self._build_tree(X[~indices_left], y[~indices_left], depth + 1)
            node.feature_index = idx
            node.threshold = thr
            node.left = left
            node.right = right

            self._accumulate_feature_importance(idx, gain)
            return node
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).astype(float)
        
        if X.ndim == 1:
            X = X.reshape(-1,1)

        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        self.root = self._build_tree(X,y,depth=0)

        total = np.sum(self.feature_importances_)
        if total > 0.0:
            self.feature_importances_ /= total

    def predict(self, X:np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        return np.array([self._predict_one(inputs) for inputs in X])
    
    def _predict_one(self, inputs: np.ndarray) -> float:
        node = self.root
        while not node.is_leaf():
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        y_pred = self.predict(X)
        # CORREÇÃO 2: Na regressão, o score é o MSE (ou a impureza), mas
        # é comum usar o R² (Coefficient of determination).
        # Manteremos como MSE, que é o padrão do seu critério.
        return mean_squared_error_manual(y, y_pred)
    
    def print_tree(self, node: Optional[DecisionTreeNode] = None, depth:int=0):
        if node is None:
            node = self.root
        indent = " " * depth
        if node.is_leaf():
            print(f"{indent}-> Folha: valor predito = {node.predicted_class:.4f}, samples = {node.num_samples}, impureza({self.criterion.upper()}): {node.gini:.4f}")
        else:
            print(f"{indent}Node:X[{node.feature_index}] < {node.threshold:.4f} (samples={node.num_samples})")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)
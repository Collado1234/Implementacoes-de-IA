import numpy as np
from collections import Counter
from typing import Optional, Tuple, Dict

class DecisionTreeNode:
    def __init__(self,
                 gini: float = None,
                 num_samples: int = None,
                 num_samples_per_class: Optional[np.ndarray] = None,
                 predicted_class: Optional[int] = None,
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

class DecisionTreeClassifier:
    def __init__(self,
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 criterion: str = 'gini',
                 min_impurity_decrease: float = 0.0):
        assert criterion in ['gini', 'entropy'], "Criterion must be 'gini' or 'entropy'"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.n_classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self.root: Optional[DecisionTreeNode] = None

    def _gini(self, y: np.ndarray) -> float:
        m = y.size
        if m == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / m
        return 1.0 - np.sum(p ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        m = y.size
        if m == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / m
        p_nonzero = p[p > 0]
        return -np.sum(p_nonzero * np.log2(p_nonzero))
    
    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None, 0.0
        
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_idx, best_thr = None, None # Inicialização com 'best_idx'

        for feature_index in range(n):
            vals = X[:, feature_index]
            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            sorted_y = y[sorted_idx]
            
            for i in range(1,m):
                if sorted_vals[i] == sorted_vals [i - 1]:
                    continue
                thr = (sorted_vals[i] + sorted_vals[i - 1]) / 2.0

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
    

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode: 
        num_samples_per_class = np.bincount(y, minlength=self.n_classes_)
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(
            gini=self._gini(y) if self.criterion == 'gini' else self._entropy(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class
        )

        #condicao de parada
        if(self.max_depth is not None and depth >= self.max_depth) or y.size < self.min_samples_split or np.unique(y).size == 1:
            return node
        
        idx, thr, gain = self._best_split(X, y)

        if idx is None or gain <= self.min_impurity_decrease:
            return node
        
        # split and recurse
        indices_left = X[:, idx] < thr
        left = self._build_tree(X[indices_left], y[indices_left], depth + 1)
        right = self._build_tree(X[~indices_left], y[~indices_left], depth + 1)
        node.feature_index = idx
        node.threshold = thr
        node.left = left
        node.right = right

        self._accumulate_feature_importance(idx, gain)

        return node

    def _accumulate_feature_importance(self, idx: int, gain: float):
        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        self.feature_importances_[idx] += gain
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_classes_ = int(np.max(y)+ 1)
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        self.root = self._build_tree(X, y, depth=0)

        total = np.sum(self.feature_importances_)
        if total > 0.0:
            self.feature_importances_ /= total
        
    def _predict_one(self, inputs: np.ndarray) -> int:
        node = self.root
        while node.left or node.right:
            if node.feature_index is None:
                break
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
            if node is None:
                break
        return node.predicted_class
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        return np.array([self._predict_one(inputs) for inputs in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probs = []
        
        for x in X:
            node = self.root
            while node.left or node.right:
                if node.feature_index is None:
                    break
                if x[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
                if node is None:
                    break
            counts = node.num_samples_per_class
            probs.append(counts / np.sum(counts))
        return np.vstack(probs)
    
    def score(self,X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def print_tree(self, node: Optional[DecisionTreeNode] = None, depth: int = 0):
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.feature_index is None:
            print(f"{indent}Leaf: predict={node.predicted_class}, samples={node.num_samples}, class_counts={node.num_samples_per_class}")
        else:
            print(f"{indent}Node: X[{node.feature_index}] < {node.threshold:.4f}  (samples={node.num_samples})")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    def explain_prediction(self, x, feature_names):
        node = self.root
        path = []
        while not node.is_leaf():
            feat_name = feature_names[node.feature_index]
            thresh = node.threshold
            value = x[node.feature_index]
            if value <= thresh:
                path.append(f"{feat_name} ({value:.2f}) <= {thresh:.2f} → esquerda")
                node = node.left
            else:
                path.append(f"{feat_name} ({value:.2f}) > {thresh:.2f} → direita")
                node = node.right
        path.append(f"Folha: classe predita = {node.predicted_class}")
        return path
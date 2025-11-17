from decision_tree_classifier import DecisionTreeClassifier
import numpy as np

class RandomForestClassifier:
    # The RandomForest Algorithm use multiple Decision Trees to improve classification accuracy. It uses bagging (bootstrap aggregating) to create diverse trees and combines their predictions.
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2,max_features='sqrt', criterion='gini'):
        #n_trees: Number of trees in the forest.
        #max_depth: Maximum depth of each tree.
        #min_samples_split: Minimum number of samples required to split an internal node.
        #max_features: Number of features to consider when looking for the best split.
        #criterion: The function to measure the quality of a split.
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.trees = []
        self.n_features_ = None        
        self.features_per_split = None

    def _get_bootstrap_sample(self, X, y):
        """
        Makes a dataset subset using bootstrap sampling (using reposition).
        
        Explanation:
        The bagging strategy consists of lessening the variance of the model by training each tree on a slightly different dataset. This is done by randomly sampling the original dataset with replacement to create multiple bootstrap samples. Each tree is then trained on one of these samples, which helps to ensure that the trees are decorrelated and improves the overall performance of the Random Forest.
        
        """
        n_samples = X.shape[0]
        # Generate random indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Return the bootstrap sample correpsponding to the generated indices
        return X[indices], y[indices]
    
    def _get_feature_subset_size(self, n_total_features):
        """
        Calculate the number of features to consider at each split.

        Explanation:
        The features randomization (Random Subspace) ensures that the trees are even more independent from each other, preventing one or two very strong features from dominating all trees (reducing correlation).        
        """
        if self.max_features == 'sqrt':
            # Common choice for classification tasks
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            # Another common choice
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            # Use the specified number of features
            return self.max_features
        else:
            # Use all features, but this is not recommended because it reduces randomness
            return n_total_features
        
    
    # random_forest.py (continuação da classe RandomForestClassifier)


    def fit(self, X, y):
        """
        Train the Random Forest. Creates and trains n_trees trees independently.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Label vector.
        """

        self.features = X.shape[1]
        self.n_features_per_split = self._get_feature_subset_size(self.features)
        self.trees = []  # Ensure the list of trees is empty before starting

        for _ in range(self.n_trees):
            X_sample, y_sample = self._get_bootstrap_sample(X, y) #bootstrap sample

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.n_features_per_split # <--- Feature Randomization
            )

            tree.fit(X_sample, y_sample) #train the tree in the bootstrap sample
            self.trees.append(tree) #store the trained tree
        
        print(f"Treinamento concluído. {self.n_trees} árvores treinadas.")
    
    
    def predict(self, X):
        """
        Make predictions by aggregating the predictions of all trees.

        Parameters:
        X (ndarray): Feature matrix for prediction.

        Returns:
        ndarray: The vector of predicted classes (by majority vote).        
        """

        #get predictions from each tree and aggregate them by majority vote
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # transpose to have shape (n_samples, n_trees)
        predictions = predictions.T
        
        # 2. Majority vote (most common class) for each sample
        y_pred = np.array([self._get_most_common_label(sample_preds) 
                           for sample_preds in predictions])
        
        return y_pred

    def _get_most_common_label(self, y_vector):
        """
        Find the most common class label in y_vector.
        """
        from collections import Counter
        # return the most common pair counted by the function 
        most_common = Counter(y_vector).most_common(1)
        return most_common[0][0]
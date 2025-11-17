import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

def _gradient(X, Y, theta):
    m = len(Y)
    return 2/m * X.T.dot(X.dot(theta) - Y)

class BaseLinearModel(ABC):
    @abstractmethod
    @classmethod
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
    
    @classmethod
    @abstractmethod
    def fit(self, X, y):
        pass

class LinearRegression2D(BaseLinearModel):
    def __init__(self):
        super().__init__()
    
    def fit(self, X : List, y : List):
        X_vector = np.array(X)
        y_vector = np.array(y)
        
        Beta = sum((X_vector - np.mean(X_vector)) * (y_vector - np.mean))/sum((X_vector - np.mean(X_vector))**2)
        
        Alpha = np.mean(y_vector) - Beta * np.mean(X_vector)

        self.intercept_ = Alpha
        self.coef_ = Beta

class ExactLinearRegressionND(BaseLinearModel):
    def __init__(self):
        super().__init__()

    def fit(self, X, Y):
        Xb = np.c_[np.ones((X.shape[0],1)), X]
        self.theta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(Y) #Formula: (X^T * X)^-1 * X^T * Y
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

class GDBRegression(BaseLinearModel):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    def fit(self, X,Y):
        Xb = np.c_[np.ones((X.shape[0],1)),X]
        m, n = Xb.shape
        print(m,n)
        self.theta = np.zeros(n)

        for _ in range(self.n_iterations):
            gradients = _gradient(Xb, Y, self.theta)
            self.theta = self.theta - self.learning_rate * gradients
        
        self.intercept = self.theta[0]
        self.coef_ = self.theta[1:]

class RidgeRegression(BaseLinearModel):
    def _init__(self, lambda_rate=1.0, learning_rate=0.01, n_iterations=1000):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, Y):
        Xb = np.c_[np.ones((X.shape[0],1)), X]
        m, n = Xb.shape
        self.theta = np.zeros(n)

        for _ in range(self.n_iterations):
            gradients = _gradient(Xb, Y, self.theta) + (2 * (self.lambda_rate / m) * self.theta)
            gradients[0] -= (2 * (self.lambda_rate / m) * self.theta[0]) #No regularization for intercept
            self.theta = self.theta - self.learning_rate * gradients
        
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

class LassoRegression(BaseLinearModel):
    def __init__(self, lambda_rate=1.0, learning_rate=0.01, n_iterations=1000):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, Y):
        Xb = np.c_[np.ones(X.shape[0],1),X]
        m , n = Xb.shape
        self.theta = np.zeros(n)
        for _ in range(self.n_iterations):
            gradients = _gradient(Xb,Y,self.theta) + (self.lambda_rate/m) * np.sign(self.theta)
            gradients[0] -= (self.lambda_rate/m) * np.sign(self.theta[0]) #No regularization for intercept
            self.theta = self.theta - self.learning_rate * gradients
        
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
    
class ElasticNetRegression(BaseLinearModel):
    def __init__(self, lambda_rate=1.0, alpha_rate=0.5, learning_rate=0.01, n_iterations=1000):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.alpha = alpha_rate
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self,X,y):
        Xb = np.c_[np.ones((X.shape[0],1)),X]
        m,n = Xb.shape
        self.theta = np.zeros(n)

        coefA = self.lambda_rate * self.alpha_rate     # L1 Regularization
        coefB = self.lambda_rate * (1 - self.alpha_rate)  # L2 Regularization

        for _ in range (self.n_iterations):
            gradients = _gradient(Xb, y, self.theta)
            gradients[1:] += (coefA / m) * np.sign(self.theta[1:]) + (2 * coefB / m) * self.theta[1:]
            self.theta -= self.learning_rate * gradients
        
        self.intercept = self.theta[0]
        self.coefficients = self.theta[1:]


class MyLinearRegresssion:
    def __init__(self, method='exact', **kwargs):
        if method == '2d':
            self.model = LinearRegression2D()
        elif method == 'exact':
            self.model = ExactLinearRegressionND()
        elif method == 'gd':
            self.model = GDBRegression(learning_rate=kwargs.get('learning_rate', 0.01), n_iterations=kwargs.get('n_iterations', 1000))
        elif method == 'ridge':
            self.model = RidgeRegression(lambda_rate=kwargs.get('lambda_rate', 1.0), learning_rate=kwargs.get('learning_rate', 0.01), n_iterations=kwargs.get('n_iterations', 1000))
        elif method == 'lasso':
            self.model = LassoRegression(lambda_rate=kwargs.get('lambda_rate', 1.0), learning_rate=kwargs.get('learning_rate', 0.01), n_iterations=kwargs.get('n_iterations', 1000))
        elif method == 'elasticnet':
            self.model = ElasticNetRegression(lambda_rate=kwargs.get('lambda_rate', 1.0), alpha_rate=kwargs.get('alpha_rate', 0.5), learning_rate=kwargs.get('learning_rate', 0.01), n_iterations=kwargs.get('n_iterations', 1000))
        else:
            raise ValueError("Invalid method specified.")
        pass

    def fit(self, X, y):
        self.model.fit(X,Y)
        self.intercept_ = self.model.intercept_
        self.coefic_ = self.model.coef_
    
    def predict(self, X):
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        return Xb.dot(np.r_[self.intercept_, self.coef_])
    
    def score(self, X, Y):
        Y_pred = self.predict(X)
        ss_total = np.sum((Y - np.mean(Y))**2)
        ss_residual = np.sum((Y - Y_pred)**2)

        return 1 - (ss_residual / ss_total)
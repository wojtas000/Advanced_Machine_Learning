import numpy as np

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def add_columns_with_combinations(X, comb_matrix, func = np.add):
    """Add columns with combinations of features."""
    for comb in comb_matrix:
        X = np.c_[X, func(X[:, comb[0]], X[:, comb[1]])]
    return X

    
def irls(X, y, max_iterations=100, tolerance=1e-6, combination_matrix = None):
    """
    Iteratively Reweighted Least Squares (IRLS) algorithm for logistic regression.
    
    Parameters:
    X (array-like, shape=(n_samples, n_features)): Training data.
    y (array-like, shape=(n_samples,)): Target labels.
    max_iterations (int): Maximum number of iterations. Default is 100.
    tolerance (float): Convergence criterion. Default is 1e-6.
    combination_matrix (array-like, shape=(n_combinations, 2)): Matrix with combinations of features.
    
    Returns:
    beta (array-like, shape=(n_features,)): Estimated coefficients.
    """
    # Add columns with combinations of features

    # ones = np.ones((X.shape[0], 1))
    # X = np.hstack((ones, X))
    
    if combination_matrix is not None:
        X = add_columns_with_combinations(X, combination_matrix)

    # Initialize beta with zeros
    beta = np.zeros(X.shape[1])
    

    print(X.shape)
    print(beta.shape)
    
    # Loop until convergence or maximum iterations reached
    for i in range(max_iterations):
        # Compute probabilities using current coefficients
        p = sigmoid(X @ beta)
        
        # Compute diagonal weight matrix
        W = np.diag(p * (1 - p))
        
        # Compute gradient and Hessian
        gradient = X.T @ (y - p)
        hessian = -X.T @ W @ X
        
        if np.linalg.det(hessian) == 0:
            break
        # Update coefficients
        beta_new = beta - np.linalg.inv(hessian) @ gradient
        
        # Check convergence
        if i>1 and np.max(np.abs(beta_new - beta)) < tolerance:
            break
        
        beta = beta_new
        
    return beta


class logistic_regression():
    
    def __init__(self):
        self.beta = None        


    def fit(self, X, y, max_iterations=100, tolerance=1e-6, combination_matrix = None):
        """
        Fit the model according to the given training data.
        
        Parameters:
        X (array-like, shape=(n_samples, n_features)): Training data.
        y (array-like, shape=(n_samples,)): Target labels.
        max_iterations (int): Maximum number of iterations. Default is 100.
        tolerance (float): Convergence criterion. Default is 1e-6.
        combination_matrix (array-like, shape=(n_combinations, 2)): Matrix with combinations of features.
        
        Returns:
        self
        """
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        self.beta = irls(X, y, max_iterations, tolerance, combination_matrix)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X (array-like, shape=(n_samples, n_features)): Samples.
        
        Returns:
        y (array-like, shape=(n_samples,)): Predicted class labels.
        """
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        return np.round(sigmoid(X @ self.beta))
    
    def accuracy(self, Xtest, ytest):
        prediction = self.predict(Xtest)
        return np.sum(prediction == ytest)/ytest.size

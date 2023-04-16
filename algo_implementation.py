import numpy as np
import copy

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def add_columns_with_combinations(X, comb_matrix, func = np.divide):
    """Add columns with combinations of features."""
    X_copy = copy.deepcopy(X)
    func_dict = {np.add: '+', np.subtract: '-', np.multiply: '*', np.divide: '/'}
    cols = X_copy.columns
    for comb in comb_matrix:
        X_copy[f'{comb[0]}{func_dict[func]}{comb[1]}'] = func(X_copy.loc[:,comb[0]], X_copy.loc[:,comb[1]])
    return X_copy

def soft_threshold(x, threshold):
    """
    Soft thresholding function
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    
def irls(X, y, regularization='none', C=0, max_iterations=100, tolerance=1e-6, combination_matrix = None):
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
        return X

    # Initialize beta with zeros
    beta = np.zeros(X.shape[1])
    
    # Loop until convergence or maximum iterations reached
    for i in range(max_iterations):
        # Compute probabilities using current coefficients
        p = sigmoid(X @ beta)
        
        # Compute diagonal weight matrix
        W = np.diag(p * (1 - p))
        
        # Compute gradient and Hessian
        gradient = X.T @ (y - p)
        hessian = -X.T @ W @ X

        # Update gradient and Hessian based on regularization
        
        if regularization == 'l2':
            update = np.insert(beta[1:], 0, 0)
            hessian_update = np.eye(X.shape[1])
            hessian_update[0][0] = 0
            gradient -= C * update
            hessian -= C * hessian_update
        
        # If Hessian is invertible stop algorithm
        if np.linalg.det(hessian) == 0:
            break
        
        # Update coefficients
        beta_new = beta - np.linalg.inv(hessian) @ gradient
        
        # Apply soft-tresholding if we use L1 regularization 
        if regularization == 'l1':
            beta_new[1:] = soft_threshold(beta_new[1:], C)
        
        # Check convergence
        if i>1 and np.max(np.abs(beta_new - beta)) < tolerance:
            break
        
        beta = beta_new
        
    return beta


class logistic_regression():
    
    def __init__(self, regularization='none', C=0, fit_intercept=True):
        """
        Parameters:
        regularization : {'none', 'l1', 'l2'}, default='none'. Regularization method to apply
        C : default=0. Strength of regularization penalty  
        fit_intercept : default=True. Specify whether a column with ones (for the intercept coefficient) should be added to dataframe.
        """
        self.beta = None        
        self.intercept_ = None
        self.coef_ = None
        self.regularization = regularization
        self.C = C
        self.fit_intercept = fit_intercept

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
        if self.fit_intercept==True:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))
        
        self.beta = irls(X, y, self.regularization, self.C, max_iterations, tolerance, combination_matrix)
        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X (array-like, shape=(n_samples, n_features)): Samples.
        
        Returns:
        y (array-like, shape=(n_samples,)): Predicted class labels.
        """
        if self.fit_intercept==True:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))
        
        return np.round(sigmoid(X @ self.beta))
    
    def accuracy(self, Xtest, ytest):
        """
        Calculate accuracy of the model 

        Parameters:
        Xtest: Testing data
        ytest: Testing labels
        """
        prediction = self.predict(Xtest)
        return np.sum(prediction == ytest)/ytest.size

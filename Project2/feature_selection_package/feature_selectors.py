import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression

# CLASSES


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on correlation with target.
    """

    def __init__(self, n_features=10):
        """
        Args:
            n_features (int): Number of features to select.
        """
        self.n_features = n_features
        self.features = None
        self.support_ = None
    
    def fit(self, X, y):
        X = pd.DataFrame(X)
        X.reset_index(drop=True, inplace=True)
        corr = pd.DataFrame(X).corrwith(pd.Series(y))
        self.features = corr.abs().sort_values(ascending=False).iloc[:self.n_features].index.to_list()
        self.support_ = np.array([True if feature in self.features else False for feature in X.columns])

        return self
    
    def transform(self, X):
        return pd.DataFrame(X)[self.features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class KbestSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on correlation with target.
    """

    def __init__(self, n_features=10):
        """
        Args:
            n_features (int): Number of features to select.
        """
        self.n_features = n_features
        self.features = None
        self.support_ = None
    
    def fit(self, X, y):
        X = pd.DataFrame(X)
        X.reset_index(drop=True, inplace=True)
        kbest = SelectKBest(k=self.n_features)
        kbest.fit(X, y)
        self.features = X.columns[kbest.get_support()]
        self.support_ = np.array([True if feature in self.features else False for feature in X.columns])

        return self
    
    def transform(self, X):
        return pd.DataFrame(X)[self.features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class MutualInformationSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on mutual information with target.
    """
    def __init__(self, n_features=10):
        """
        Args:
            n_features (int): Number of features to select.
        """

        self.n_features = n_features
        self.features = None
        self.support_ = None
    
    def fit(self, X, y):
        X = pd.DataFrame(X)
        X.reset_index(drop=True, inplace=True)
        self.features = mutual_info_classif(X, y).argsort()[-self.n_features:][::-1]
        self.support_ = np.array([True if feature in self.features else False for feature in X.columns])
        return self
    
    def transform(self, X):
        return pd.DataFrame(X)[self.features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class RandomForestSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on feature importance from random forest.
    """

    def __init__(self, n_features=10, threshold=None):
        """
        Args:
            n_features (int): Number of features to select.
            threshold (float): Threshold for feature importance.
        """

        self.n_features = n_features
        self.threshold = threshold
        self.features = None
        self.support_ = None
    
    def fit(self, X, y):
        X = pd.DataFrame(X)
        X.reset_index(drop=True, inplace=True)
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        clf.fit(X, y)
        importances = pd.Series(clf.feature_importances_)
        if self.threshold is not None:
            self.features = self.high_importance_features(importances, self.threshold)
        else:
            self.features = self.top_n_features(importances, self.n_features)
        self.support_ = np.array([True if feature in self.features else False for feature in X.columns])

        return self
    
    def transform(self, X):
        return pd.DataFrame(X)[self.features]
    
    
    def top_n_features(self, importances, n):
        return importances.sort_values(ascending=False).iloc[:n].index

    def high_importance_features(self, importances, threshold):
        return importances[importances > threshold].index
    
    
class Debug(BaseEstimator, TransformerMixin):
    """
    Transformer that gives intermediate access to shape of data.
    """

    def transform(self, X):
        self.shape = X.shape
        return X

    def fit(self, X, y=None):
        return self


class EnsembleSelector(BaseEstimator, TransformerMixin):
    """
    Ensemble feature selector.
    """

    def __init__(self, selectors):
        """
        Args:
            selectors (list): List of feature selectors.
        """

        self.selectors = selectors
        self.features = None
        self.support_ = None
    
    def fit(self, X, y):
        self.features = self.select_features(X, y)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.features] = True
        return self
    
    def transform(self, X):
        return pd.DataFrame(X)[self.features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def select_features(self, X, y):
        total_support = np.zeros(X.shape[1], dtype=int)
        for selector in self.selectors:
            selector.fit(X, y)
            if isinstance(selector, SelectKBest):
                support = selector.get_support()
            else:
                support = selector.support_
        
            total_support += support.astype(int)
        threshold = len(self.selectors) // 2 + 1
        features = np.where(total_support >= threshold)[0]
        self.support_ = (total_support >= threshold)
        return features



if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    selector = MutualInformationSelector(n_features=5)
    selector.fit(X, y)
    print(selector.features)
    print(selector.support_)
    
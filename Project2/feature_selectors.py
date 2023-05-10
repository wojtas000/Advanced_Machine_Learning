import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer


# CLASSES


class CorrelationSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.features = None
    
    def fit(self, X, y):
        corr = pd.DataFrame(X).corrwith(pd.Series(y))
        self.features = corr.abs().sort_values(ascending=False).iloc[:self.n_features].index
        return self
    
    def transform(self, X):
        return pd.DataFrame(X)[self.features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class MutualInformationSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_features=10):
        self.n_features = n_features
        self.features = None
    
    def fit(self, X, y):
        self.features = mutual_info_classif(X, y).argsort()[-self.n_features:][::-1]
        return self
    
    def transform(self, X):
        return pd.DataFrame(X)[self.features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class RandomForestSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10, threshold=None):
        self.n_features = n_features
        self.threshold = threshold
        self.features = None
    
    def fit(self, X, y):
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        clf.fit(X, y)
        importances = pd.Series(clf.feature_importances_)
        if self.threshold is not None:
            self.features = self.high_importance_features(importances, self.threshold)
        else:
            self.features = self.top_n_features(importances, self.n_features)

        return self
    
    def transform(self, X):
        return pd.DataFrame(X)[self.features]
    
    
    def top_n_features(self, importances, n):
        return importances.sort_values(ascending=False).iloc[:n].index

    def high_importance_features(self, importances, threshold):
        return importances[importances > threshold].index
    
    
class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        self.shape = X.shape
        return X

    def fit(self, X, y=None):
        return self

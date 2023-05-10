import numpy as np
import pandas as pd
from copy import copy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from feature_selectors import Debug, RandomForestSelector


def performance_score(accuracy, n_features):
    return np.round(accuracy - 0.01 * np.abs(0.01*n_features - 1), 2)


def single_evaluation(X_train, y_train, X_val, y_val, selector=PCA(n_components=5), classifier=RandomForestClassifier(), scaler=MinMaxScaler()):
    
    pipeline = make_pipeline(scaler, selector, Debug(), classifier)
    pipeline.fit(X_train, y_train) 
    accuracy = pipeline.score(X_val, y_val)
    n_features = pipeline.steps[-2][1].shape[1]
    perf_score = performance_score(accuracy, n_features)
    return accuracy, perf_score, n_features


def full_evaluation(X_train, y_train, X_val, y_val, selectors, classifiers, n_features):
    
    df = pd.DataFrame(columns=['Selector', 'Classifier', 'Number_of_Features', 'Accuracy', 'Performance_score'])

    for selector in selectors:
        for classifier in classifiers:
            for n in n_features:
                if isinstance(selector, PCA):
                    selector.n_components = n
                else:
                    selector.n_features = n
                accuracy, perf_score, _ = single_evaluation(X_train, y_train, X_val, y_val, selector, classifier)

                if isinstance(classifier, SVC):
                    classifier_name = classifier.__class__.__name__ + '_' + classifier.kernel
                else:
                    classifier_name = classifier.__class__.__name__

                df = pd.concat([df, pd.DataFrame([[selector.__class__.__name__, classifier_name, n, accuracy, perf_score]], columns=['Selector', 'Classifier', 'Number_of_Features', 'Accuracy', 'Performance_score'])])

    return df



if __name__=="__main__":
    
    X, y = load_breast_cancer(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_train, X_valid = X.iloc[:400], X.iloc[400:]
    y_train, y_valid = y.iloc[:400], y.iloc[400:]
    selectors = [PCA()]
    classifiers = [RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)]
    n_features = [2, 3, 4, 5]
    df = full_evaluation(X_train, y_train, X_valid, y_valid, selectors, classifiers, n_features)
    print(df)

    selector = RandomForestSelector()
    classifier = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    score, perf_score = single_evaluation(X_train, y_train, X_valid, y_valid, selector, classifier)
    print(score, perf_score)

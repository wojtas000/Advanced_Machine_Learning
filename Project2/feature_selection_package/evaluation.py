import numpy as np
import pandas as pd
from copy import copy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, FeatureUnion, FunctionTransformer
from feature_selection_package.feature_selectors import Debug, RandomForestSelector, CorrelationSelector
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score


def performance_score(accuracy, n_features, dataset_type=None):
    """
    Compute performance score.
    Args:
        accuracy (float): Accuracy of the model.
        n_features (int): Number of features.
    Returns:
        float: Performance score.
    """
    if dataset_type == 'artificial':
        return np.round(accuracy - 0.01 * max(0, 0.2*n_features - 1), 4)
    else:
        return np.round(accuracy - 0.01 * max(0, 0.01*n_features - 1), 4)


def feature_selection(X, y, selectors=PCA(n_components=5), scaler=MinMaxScaler()):
    """
    Prepare feature selection pipeline and return supported features.
    Args:
        X (pd.DataFrame): Data.
        y (pd.Series): Labels.
        selectors: List of feature selectors (or single feature selector).
        scaler (object): Scaler.
    Returns:
        pipeline (object): Feature selection pipeline.
        supported_features (np.array): Array of supported features.
    """
    if isinstance(selectors, list):
        pipeline = make_pipeline(scaler, *selectors)
    else:
        pipeline = make_pipeline(scaler, selectors)
    pipeline.fit_transform(X, y)
    feature_selection_model = pipeline.steps[-1][-1]
    if isinstance(feature_selection_model, SelectKBest):
        supported_features = feature_selection_model.get_support()
    elif isinstance(feature_selection_model, PCA):
        supported_features = np.ones(X.shape[1], dtype=bool)
    else:
        supported_features = feature_selection_model.support_
   
    return pipeline, supported_features


def single_evaluation(X_train, y_train, X_val, y_val, feature_selection_pipeline, classifier, dataset_type='artificial'):
    """
    Evaluate single combination of selector and classifier.
    Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation data.
        y_val (pd.Series): Validation labels.
        selector (object): Feature selector.
        classifier (object): Classifier.
    Returns:
        float: Accuracy of the model.
        float: Performance score.
        int: Number of features selected.
    """
    transformer = FunctionTransformer(feature_selection_pipeline.transform)
    pipeline = make_pipeline(transformer, Debug(), classifier)
    pipeline.fit(X_train, y_train)

    # predict validation data
    y_pred = pipeline.predict(X_val)

    # get number of features
    n_features = pipeline.steps[-2][1].shape[1]
    
    # compute performance score and accuracy
    balanced_accuracy = balanced_accuracy_score(y_val, y_pred) 
    perf_score = performance_score(balanced_accuracy, n_features, dataset_type)


    return balanced_accuracy, perf_score, n_features


def full_evaluation(X_train, y_train, X_val, y_val, selectors, classifiers, dataset_type):
    """
    Evaluate all combinations of selectors and classifiers.
    Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation data.
        y_val (pd.Dataframe): Validation labels.
        selectors (list): List of feature selectors.
        classifiers (list): List of classifiers.
        n_features (list): List of numbers of features.
    Returns:
        df (pd.Dataframe): Dataframe containing evaluation metrics of the selector-classifier combinations.

    """
    df = pd.DataFrame(columns=['Selector', 'Classifier', 'Number_of_Features', 'Accuracy', 'Performance_score', 'Supported_Features'])

    for selector in selectors:
        feature_selection_pipeline, supported_features = feature_selection(X_train, y_train, selector)
        for classifier in classifiers:
            accuracy, perf_score, n_features = single_evaluation(X_train, y_train, X_val, y_val, feature_selection_pipeline, classifier, dataset_type)
            
            if isinstance(selector, SelectKBest):
                selector_name = selector.__class__.__name__ + '_' + selector.score_func.__name__
            else:
                selector_name = selector.__class__.__name__
            
            if isinstance(classifier, SVC):
                classifier_name = classifier.__class__.__name__ + '_' + classifier.kernel
            else:
                classifier_name = classifier.__class__.__name__
            
            df = pd.concat([df, pd.DataFrame([[selector_name, classifier_name, n_features, accuracy, perf_score, supported_features]], columns=['Selector', 'Classifier', 'Number_of_Features', 'Accuracy', 'Performance_score', 'Supported_Features'])])

    return df



if __name__=="__main__":
    
    X, y = load_breast_cancer(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_train, X_valid = X.iloc[:400], X.iloc[400:]
    y_train, y_valid = y.iloc[:400], y.iloc[400:]
    selectors = [CorrelationSelector(n_features=5)]
    classifiers = [RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), SVC(kernel='linear', random_state=0), SVC(kernel='rbf', random_state=0)]
    n_features = [2, 3, 4, 5]
    results = pd.DataFrame(columns=['Selector', 'Classifier', 'Number_of_Features', 'Accuracy', 'Performance_score'])
    for n in n_features:
        selectors = [CorrelationSelector(n_features=n)]
        df = full_evaluation(X_train, y_train, X_valid, y_valid, selectors, classifiers)
        results = pd.concat([results, df])
    print(results)

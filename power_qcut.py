
import itertools
from functools import reduce
from typing import Callable, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from IPython.display import clear_output, display
from sklearn import metrics, tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm


def map_operation(feature_1, feature_2, operations):
    ''' Map operations to features'''
    to_return = set()
    for operation in operations:
        left_side = f'{feature_1} {operation} {feature_2}'
        right_side = f'{feature_2} {operation} {feature_1}'
        if operation in ['+', '*', '-']:
            to_return.update([left_side])
        else:
            to_return.update([left_side, right_side])
    return to_return


def prepare_cols_comb(cols, operations=None):
    """ Get all 2 element combinations from given columns
    
    Args:
        cols: list of columns
        operations: list of operations to use, default ['+', '-', '*', '/']
    Returns:
        list of all combinations of columns with operations
    """
    if operations is None:
        operations = ['+', '-', '*', '/']
    combs = itertools.combinations(cols, 2)
    mapped_cols = map(lambda x: map_operation(x[0], x[1], operations), combs)
    mapped_cols = list(reduce(lambda s1, s2: s1.union(s2), mapped_cols))
    return np.union1d(mapped_cols, cols)

def qcut_iterations(col, data, y_col, score_function, n_bins):
    '''
    Function to calculate the monotonicity of a variable
    '''
    try:
        if isinstance(col, str):
            qcut_vals = data.groupby(pd.qcut(data.eval(col), n_bins, duplicates='drop'))[y_col].agg(score_function)
            ref_x = list(range(1, n_bins+1))
            temp_array = np.array([ref_x, [1]*n_bins])
            slope = sm.OLS(qcut_vals, temp_array.T).fit().params.x1
            mono = scipy.stats.spearmanr(ref_x, qcut_vals)
            n_features = len(col.split(' '))
            n_features = n_features - n_features//2
            return [col, n_features, slope, mono[0],
                                qcut_vals.min(), qcut_vals.max(),
                                qcut_vals[qcut_vals>0].sum(), qcut_vals[qcut_vals<0].sum()
                       ]
        if isinstance(col, list):
            print(col, 'is a list')
    except ValueError as errors:
        print(col, errors)
    except IndexError as errors:
        print(col, errors)
    return []

def qcut_selection(data: pd.DataFrame, y_col: str, cols_to_check: List[str], score_function: Callable,
                   n_bins: int) -> pd.DataFrame:
    """ Returns dataframe with features and their parameters for monotonicity check, such as slope and spearman correlation, due to qcut method.
    
    Args:
        data: dataframe with data
        y_col: name of column with target variable
        cols_to_check: list of columns to check
        score_function: function to calculate score for each bin
        n_bins: number of bins to use
    Returns:
        dataframe with features and their parameters for monotonicity check
    """
    results = []
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(qcut_iterations)(item, data, y_col, score_function, n_bins)
                                         for item in tqdm(cols_to_check))
    return_df = pd.DataFrame([r for r in results if len(r)], columns = ['feature', 'n_features', 'slope', 'mono', 'min_val', 'max_val', 'sum_pos', 'sum_neg'])
    return_df = return_df.assign(min_max_ratio = lambda x: x.max_val/abs(x.min_val))
    return return_df.sort_values('mono')

# def cut_iterations(col, data, y_col, score_function, n_bins):
#     '''
#     Function to calculate the monotonicity of a variable
#     '''
#     try:
#         if isinstance(col, str):
#             qcut_vals = data.groupby(pd.cut(data.eval(col), n_bins))[y_col].agg(score_function)
#             ref_x = list(range(1, n_bins+1))
#             temp_array = np.array([ref_x, [1]*n_bins])
#             slope = sm.OLS(qcut_vals, temp_array.T).fit().params.x1
#             mono = scipy.stats.spearmanr(ref_x, qcut_vals)
#             n_features = len(col.split(' '))
#             n_features = n_features - n_features//2
#             return [col, n_features, slope, mono[0],
#                                 qcut_vals.min(), qcut_vals.max(),
#                                 qcut_vals[qcut_vals>0].sum(), qcut_vals[qcut_vals<0].sum()
#                        ]
#         if isinstance(col, list):
#             print(col, 'is a list')
#     except ValueError as errors:
#         print(col, errors)
#     except IndexError as errors:
#         print(col, errors)
#     return []

# def cut_selection(data: pd.DataFrame, y_col: str, cols_to_check: List[str], score_function: Callable,
#                   n_bins: int) -> pd.DataFrame:
#     """ Returns dataframe with features and their parameters for monotonicity check, such as slope and spearman correlation, due to cut method.
        
#     Args:
#         data: dataframe with data
#         y_col: name of column with target variable
#         cols_to_check: list of columns to check
#         score_function: function to calculate score for each bin
#         n_bins: number of bins to use
#     Returns:
#         dataframe with features and their parameters for monotonicity check
#     """
#     results = []
#     results = joblib.Parallel(n_jobs=-1)(joblib.delayed(cut_iterations)(item, data, y_col, score_function, n_bins)
#                                          for item in tqdm(cols_to_check))
#     return_df = pd.DataFrame([r for r in results if len(r)], columns = ['feature', 'n_features', 'slope', 'mono', 'min_val', 'max_val', 'sum_pos', 'sum_neg'])
#     return_df = return_df.assign(min_max_ratio = lambda x: x.max_val/abs(x.min_val))
#     return return_df.sort_values('mono')


def common_member(a_list, b_list):
    """ Returns common elements from two lists
    Args:
        a_list: list
        b: list
    Returns:
        list with common elements
    """
    a_set = set(a_list)
    b_set = set(b_list)

    if a_set & b_set:
        return a_set & b_set
    return []


def qcut_fold_validation(data_set: pd.DataFrame, k_number: int, y_col: str, cols_to_check: List[str], score_function: Callable,
                         n_bins: int, condition: str = 'abs(mono) > 0.8') -> pd.DataFrame:
    """Function is used to select features for monotonicity check. It splits data into k_number 
       of parts and checks monotonicity for each part. Then it check if common feature is monotonic on whole dataset and returns it.
    Args:
        data_set: dataframe with data
        k_number: number of parts to split data
        y_col: name of column with target variable
        cols_to_check: list of columns to check
        score_function: function to calculate score for each bin
        n_bins: number of bins to use
        condition: condition to select features
    Returns:
        dataframe with features and their parameters for monotonicity check
    """
    leng = int(data_set.shape[0] / k_number)
    results = cols_to_check
    for i in range(k_number):
        clear_output()
        print('Round ', i+1, '/', k_number)
        data_frame = data_set[leng * i: leng * (i+1)]
        res = qcut_selection(data_frame,
                             y_col,
                             results,
                             score_function,
                             n_bins).query(condition)
        if res.feature.shape[0] == 0:
            clear_output()
            print('Zero features :(')
            return pd.DataFrame([])
        results = common_member(results, list(res.feature))
    clear_output()
    res = qcut_selection(data_set,
                         y_col,
                         results,
                         score_function, n_bins).query(condition)
    return res


# def cut_fold_validation(data_set: pd.DataFrame, k_number: int, y_col: str, cols_to_check: List[str], score_function: Callable,
#                         n_bins: int, condition: str = 'abs(mono) > 0.8') -> pd.DataFrame:
#     """Function is used to select features for monotonicity check. It splits data into k_number 
#        of parts and checks monotonicity for each part. Then it check if common feature is monotonic on whole dataset and returns it. 
#     Args:
#         data_set: dataframe with data
#         k_number: number of parts to split data
#         y_col: name of column with target variable
#         cols_to_check: list of columns to check
#         score_function: function to calculate score for each bin
#         n_bins: number of bins to be used
#         condition: condition to select features
#     Returns:
#         dataframe with features and their parameters for monotonicity check
#     """
#     leng = int(data_set.shape[0] / k_number)
#     results = cols_to_check
#     for i in range(k_number):
#         clear_output()
#         print('Round ', i+1, '/', k_number)
#         data_frame = data_set[leng * i: leng * (i+1)]
#         res = cut_selection(data_frame,
#                             y_col,
#                             results,
#                             score_function,
#                             n_bins).query(condition)
#         results = common_member(results, list(res.feature))
#         if res.feature.shape[0] == 0:
#             clear_output()
#             print('Zero features :(')
#             return pd.DataFrame([])
#     clear_output()
#     res = cut_selection(data_set,
#                         y_col,
#                         results,
#                         score_function, n_bins).query(condition)
#     return res
